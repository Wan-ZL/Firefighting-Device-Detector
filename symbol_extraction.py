'''
Project     : SimCLR-leftthomas 
File        : symbol_extraction.py
Author      : Zelin Wan
Date        : 4/18/24
Description : Get the annotation from a JSON file and extract symbols from the annotation.
'''
import hashlib
import json
import os
import cv2

def extract_symbols(data_path, json_file):
    '''
    Get the annotation from a JSON file and extract symbols from the annotation.
    '''
    # Load the JSON file
    with open(data_path + json_file, 'r') as f:
        data = json.load(f)

    # get a mapping from image_id to image_name
    image_id_to_name = {}
    for image in data['images']:
        image_id_to_name[image['id']] = image['file_name']

    # Extract the symbols from the annotation
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
        if x+w > image.shape[1]:
            w = image.shape[1] - x
        if y+h > image.shape[0]:
            h = image.shape[0] - y
        symbol = image[y:y+h, x:x+w]

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

# data_path = 'Firefighting Device Detection.v6i.coco/train/'
# json_file = '_annotations.coco.json'
# extract_symbols(data_path, json_file)




