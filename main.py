'''
Project     : SimCLR-leftthomas 
File        : main.py
Author      : Zelin Wan
Date        : 4/19/24
Description : 
'''
import os

import cv2
import torch
from PIL import Image
from roboflow import Roboflow

import utils
from model import Model
import torch.nn as nn

from train_model import train_model


def crop_rect(event, x, y, flags, param):
    global rect_endpoint_tmp, rect_endpoint, cropping, roi

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_endpoint_tmp = [(x, y)]
        cropping = True


    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            rect_endpoint_tmp[1:] = [(x, y)]
            img_tmp = picture.copy()
            cv2.rectangle(img_tmp, rect_endpoint_tmp[0], rect_endpoint_tmp[1], (0, 255, 0), 1)  # Change thickness here
            cv2.imshow("Select a symbol using mouse (press down, drag/move, release).", img_tmp)

    elif event == cv2.EVENT_LBUTTONUP:
        rect_endpoint_tmp[1:] = [(x, y)]
        cropping = False # cropping is finished

        rect_endpoint = rect_endpoint_tmp
        cv2.rectangle(picture, rect_endpoint[0], rect_endpoint[1], (0, 255, 0), 1)  # Change thickness here
        cv2.imshow("Select a symbol using mouse (press down, drag/move, release).", picture)

        if len(rect_endpoint) == 2:
            roi = picture[rect_endpoint[0][1]:rect_endpoint[1][1], rect_endpoint[0][0]:rect_endpoint[1][0]]
            cv2.imshow("Press 'c' to continue", roi)
            cv2.waitKey(1)


if __name__ == '__main__':
    # ask if user want to use pretrained model or train a new model
    pretrained_model = input("Do you want to use a pretrained model? Suggest to press 'y' for the first time. (y/n): ")
    if pretrained_model == 'y':
        model_path = 'models/pre_trained_256_0.5_200_128_500_20240420041730_model.pth'
        # download dataset from roboflow then save to local
        rf = Roboflow(api_key="eSdWlAgPzoRe5BDLTdFr")
        project = rf.workspace("yaid-pzikt").project("firefighting-device-detection")
        version = project.version(6)
        dataset = version.download("coco")
        print("Downloaded Complete")
        utils.FirefightingDataset('Firefighting-Device-Detection-6/test/', '_annotations.coco.json',
                                              transform=utils.test_transform)
        # check if the model exists
        if not os.path.exists(model_path):
            os.makedirs('models', exist_ok=True)
            print("The pretrained model does not exist. Please put the given trained model under the 'models' folder.")
            exit()
    elif pretrained_model == 'n':
        train_model()
        model_path = 'models/256_0.5_200_128_500_model.pth'
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
        exit()

    # load model.state_dict()
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_dim = 256
    model = Model(feature_dim)
    model.eval()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)

    # load a picture and a symbol
    data_path = 'Firefighting-Device-Detection-6/test/'

    picture_name = '3-2-2a_PNG.rf.441833ca66d864dc3919c9b9de8c9568.jpg'
    picture = cv2.imread(data_path + picture_name)

    use_pre_set_symbol = input("Do you want to test a pre-selected symbol (press y) or use mouse to crop a symbol (press n)? (y/n): ")
    if use_pre_set_symbol == 'y':
        # pick any file from folder '5' as a symbol using os.listdir()
        symbol_path = data_path + '5/'
        symbol_name = os.listdir(symbol_path)[0]
        symbol = Image.open(symbol_path + symbol_name)
    elif use_pre_set_symbol == 'n':
        # open the picture in cv2, then use mouse to crop a symbol. Save the cropped symbol to the variable symbol
        rect_endpoint_tmp = []
        rect_endpoint = []
        cropping = False

        # setup the mouse callback function
        cv2.namedWindow("Select a symbol using mouse (press down, drag/move, release).")
        cv2.setMouseCallback("Select a symbol using mouse (press down, drag/move, release).", crop_rect)

        # keep looping until the 'c' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("Select a symbol using mouse (press down, drag/move, release).", picture)
            key = cv2.waitKey(1) & 0xFF
            # if the 'c' key is pressed, break from the loop
            if key == ord("c"):
                break

        symbol = None
        if len(rect_endpoint) == 2:
            cv2.destroyAllWindows()
            symbol = picture[rect_endpoint[0][1]:rect_endpoint[1][1], rect_endpoint[0][0]:rect_endpoint[1][0]]
            cv2.imshow("Press 'c' to continue", symbol)
            cv2.waitKey(1)
        # transform the symbol to PIL image, then to tensor
        symbol = Image.fromarray(symbol)
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
        exit()



    # if you cannot drop a symbol, use the following code to load a symbol
    # symbol_name = '5/category_5_2dae7bbf23dcdf3841a47b9b92893c96.png'
    # symbol = Image.open(data_path + symbol_name)

    symbol = utils.test_transform(symbol).unsqueeze(dim=0).to(device)

    # find the location of a given symbol with sliding window
    stride = 8
    window_size = 32
    similarity_threshold = 0.75
    buffer_size = 128

    # # sliding window
    windows = torch.zeros(0, 3, window_size, window_size).to(device)
    positions = []
    for y in range(0, picture.shape[0] - window_size, stride):
        for x in range(0, picture.shape[1] - window_size, stride):
            window = picture[y:y + window_size, x:x + window_size]
            # convert window to PIL image, then to tensor
            window_tensor = utils.test_transform(Image.fromarray(window)).unsqueeze(dim=0).to(device)
            windows = torch.cat((windows, window_tensor), dim=0)
            positions.append((x, y))

            # If we have collected buffer_size (128) windows, process them all at once
            if len(windows) == buffer_size:
                # print("processing batch")
                # stack window_tensor together
                features, _ = model(windows)
                similarities = torch.nn.functional.cosine_similarity(features, model(symbol)[0])

                # draw rectangles around windows with high similarity
                for i, similarity in enumerate(similarities):
                    if similarity > similarity_threshold:
                        print("high similarity", similarity.item())
                        x, y = positions[i]
                        cv2.rectangle(picture, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
                        # cv2.imshow('window', windows[i].permute(1, 2, 0).cpu().numpy())
                        cv2.imshow('picture', picture)
                        cv2.waitKey(1)

                # Reset windows and positions for the next batch
                windows = torch.zeros(0, 3, window_size, window_size).to(device)
                positions = []

    # Process the remaining windows if their count is less than buffer_size (128)
    if len(windows) > 0:
        features, _ = model(windows)
        similarities = torch.nn.functional.cosine_similarity(features, model(symbol)[0])

        for i, similarity in enumerate(similarities):
            if similarity > similarity_threshold:
                print("high similarity", similarity.item())
                x, y = positions[i]
                cv2.rectangle(picture, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
                # cv2.imshow('window', windows[i].permute(1, 2, 0).cpu().numpy())
                cv2.imshow('picture', picture)
                cv2.waitKey(1)
    # # on center of cv2, show txt "all high similarity symbol found, press any key to exit." on the center of the picture
    cv2.destroyAllWindows()
    # save the picture to local
    cv2.imwrite('result.jpg', picture)

    picture = cv2.putText(picture, "Complete! Result saved as 'result.jpg'. Press any key to exit.", (int(picture.shape[1] / 2) - 400, int(picture.shape[0] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("All high similarity symbols found. Press any key to exit.", picture)
    print("found all high similarity windows")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



