import hashlib
import json
import os
import random

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class FirefightingDataset(Dataset):
    def __init__(self, data_path, json_file, transform=None):
        self.data_path = data_path
        self.json_file = json_file
        self.transform = transform
        self.symbols = self.extract_symbols(data_path, json_file)
        self.classes = list(set([symbol['category'] for symbol in self.symbols]))
        self.num_classes = len(self.classes)

        # Group symbols by category
        self.symbols_by_category = {}
        for symbol in self.symbols:
            if symbol['category'] not in self.symbols_by_category:
                self.symbols_by_category[symbol['category']] = []
            self.symbols_by_category[symbol['category']].append(symbol)

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, idx):
        symbol = self.symbols[idx]
        img1 = Image.open(symbol['file_path'])
        label = symbol['category']

        # Select another symbol from the same category
        same_category_symbols = self.symbols_by_category[label]
        symbol2 = random.choice(same_category_symbols)
        img2 = Image.open(symbol2['file_path'])

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def extract_symbols(self, data_path, json_file):
        '''
        Get the annotation from a JSON file and extract symbols from the annotation.
        '''
        print("Extracting symbols")
        # Load the JSON file
        with open(data_path + json_file, 'r') as f:
            data = json.load(f)

        # get a mapping from image_id to image_name
        image_id_to_name = {}
        for image in data['images']:
            image_id_to_name[image['id']] = image['file_name']

        # Extract the symbols from the annotation
        symbols = []
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            bbox = annotation['bbox']
            # open the image and draw cropped symbol
            image_path = data_path + image_id_to_name[image_id]
            image = cv2.imread(image_path)

            # crop the symbol
            x, y, w, h = bbox
            # convert float to int
            x, y, w, h = int(x), int(y), int(w), int(h)
            # handle the case when the bbox is out of the image
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + w > image.shape[1]:
                w = image.shape[1] - x
            if y + h > image.shape[0]:
                h = image.shape[0] - y
            symbol = image[y:y + h, x:x + w]

            # store the cropped symbol in the corresponding category folder under the data_path
            category_id = annotation['category_id']
            # create the category folder if it does not exist
            category_folder = data_path + str(category_id) + '/'
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)

            # store the symbol in the category folder with the hash value as the file name
            hash_symbol = hashlib.md5(symbol.tobytes()).hexdigest()
            symbol_path = category_folder + 'category_' + str(category_id) + '_' + hash_symbol + '.png'
            cv2.imwrite(symbol_path, symbol)
            symbols.append({'category': category_id, 'file_path': symbol_path})
        return symbols


# class FirefightingDataset(Dataset):
#     def __init__(self, data_path, json_file, transform=None):
#         self.data_path = data_path
#         self.json_file = json_file
#         self.transform = transform
#         self.symbols = self.extract_symbols(data_path, json_file)
#         self.classes = list(set([symbol['category'] for symbol in self.symbols]))
#         self.num_classes = len(self.classes)
#
#     def __len__(self):
#         return len(self.symbols)
#
#     def __getitem__(self, idx):
#         symbol = self.symbols[idx]
#         img = Image.open(symbol['file_path'])
#         img = np.array(img)
#         img = Image.fromarray(img)
#         label = symbol['category']
#
#
#         pos_1 = img
#         pos_2 = img
#         if self.transform is not None:
#             pos_1 = self.transform(img)
#             pos_2 = self.transform(img)
#
#         return pos_1, pos_2, label
#
#     def extract_symbols(self, data_path, json_file):
#         '''
#         Get the annotation from a JSON file and extract symbols from the annotation.
#         '''
#         # Load the JSON file
#         with open(data_path + json_file, 'r') as f:
#             data = json.load(f)
#
#         # get a mapping from image_id to image_name
#         image_id_to_name = {}
#         for image in data['images']:
#             image_id_to_name[image['id']] = image['file_name']
#
#         # Extract the symbols from the annotation
#         symbols = []
#         for annotation in data['annotations']:
#             image_id = annotation['image_id']
#             bbox = annotation['bbox']
#             # open the image and draw cropped symbol
#             image_path = data_path + image_id_to_name[image_id]
#             image = cv2.imread(image_path)
#
#             # crop the symbol
#             x, y, w, h = bbox
#             # convert float to int
#             x, y, w, h = int(x), int(y), int(w), int(h)
#             # handle the case when the bbox is out of the image
#             if x < 0:
#                 x = 0
#             if y < 0:
#                 y = 0
#             if x + w > image.shape[1]:
#                 w = image.shape[1] - x
#             if y + h > image.shape[0]:
#                 h = image.shape[0] - y
#             symbol = image[y:y + h, x:x + w]
#
#             # store the cropped symbol in the corresponding category folder under the data_path
#             category_id = annotation['category_id']
#             # create the category folder if it does not exist
#             category_folder = data_path + str(category_id) + '/'
#             if not os.path.exists(category_folder):
#                 os.makedirs(category_folder)
#
#             # store the symbol in the category folder with the hash value as the file name
#             hash_symbol = hashlib.md5(symbol.tobytes()).hexdigest()
#             symbol_path = category_folder + 'category_' + str(category_id) + '_' + hash_symbol + '.png'
#             cv2.imwrite(symbol_path, symbol)
#             symbols.append({'category': category_id, 'file_path': symbol_path})
#         return symbols

train_transform = transforms.Compose([
    # RandomResizedCrop: the cropped image cover 80% of the original image
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    # transforms.Resize((32, 32)),  # Resize all images to 32x32
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(360),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize all images to 32x32
    # transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(360),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
