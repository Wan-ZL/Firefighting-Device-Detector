'''
Project     : SimCLR-leftthomas 
File        : train_model.py
Author      : Zelin Wan
Date        : 4/18/24
Description : Train a SimCLR model for image similarity learning.
'''

import argparse
import os
import time

import cv2
import pandas as pd
import torch
from fsspec.compression import unzip
from sympy.physics.vector import curl
from thop import profile, clever_format
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from roboflow import Roboflow

import utils
from model import Model


def train(net, data_loader, train_optimizer, epoch=1, epochs=500, device='cpu'):
    net.train()
    total_loss, total_num = 0.0, 0
    train_bar = tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        # convert pos_1 and pos_2 to numpy then show with cv2
        pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.size(0), device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        # sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.size(0), -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

def test(net, memory_loader, test_loader, epoch=1, epochs=500, device='cpu'):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        for data, _, target in tqdm(memory_loader, desc='Feature extracting'):
            data = data.to(device)
            feature, _ = net(data)
            feature_bank.append(feature)
            feature_labels.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.cat(feature_labels, dim=0).to(device)
        test_bar = tqdm(test_loader)
        for data, _, target in test_bar:
            data = data.to(device)
            target = target.to(device)
            feature, _ = net(data)

            total_num += data.size(0)
            # compute cos similarity
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            num_class = memory_loader.dataset.num_classes
            one_hot_label = torch.zeros(data.size(0) * k, num_class, device=device)
            # [B*K, NC]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, NC]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, num_class) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100




def train_model(feature_dim=256, temperature=0.5, k=200, batch_size=128, epochs=500, num_workers=8):

    # custom batch size and epochs for testing
    # batch_size = 128
    # epochs = 30
    # num_workers = 1

    # download dataset from roboflow then save to local
    rf = Roboflow(api_key="eSdWlAgPzoRe5BDLTdFr")
    project = rf.workspace("yaid-pzikt").project("firefighting-device-detection")
    version = project.version(6)
    dataset = version.download("coco")
    print("Downloaded Complete")


    # print hyperparameters
    print("Hyperparameters: feature_dim: {}, temperature: {}, k: {}, batch_size: {}, epochs: {}, workers: {}".format(feature_dim, temperature, k, batch_size, epochs, num_workers))

    # data prepare
    train_data = utils.FirefightingDataset('Firefighting-Device-Detection-6/train/', '_annotations.coco.json',
                                           transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

    memory_data = utils.FirefightingDataset('Firefighting-Device-Detection-6/valid/', '_annotations.coco.json',
                                            transform=utils.train_transform)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    test_data = utils.FirefightingDataset('Firefighting-Device-Detection-6/test/', '_annotations.coco.json',
                                          transform=utils.test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    # model prepare
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Training on', device)

    model = Model(feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # calculate FLOPs and params
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    os.makedirs('models', exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, epochs, device)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, epoch, epochs, device)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))

        data_frame.to_csv('models/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'models/{}_model.pth'.format(save_name_pre))
            print("found new best acc, saving model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=256, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--workers', default=8, type=int, help='Number of workers for data loader')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    num_workers = args.workers

    train_model(feature_dim, temperature, k, batch_size, epochs, num_workers)




