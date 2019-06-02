#!/usr/bin/env python

import os
import sys
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

# Loop through each class  in training data folder
for folder in glob(sys.argv[1] + '/train/*/'):
    dirs = folder.strip().split('/')
    if dirs[1] != '':
        item_type = dirs[2]
    else:
        item_type = dirs[3]

    # Check for data that has already been split and skip
    completed = os.listdir(sys.argv[1] + '/val/')
    if item_type in completed:
        print('\''+item_type+'\' already found in validation folder')
        continue

    print('Creating validation folder for data in \''+folder+'\'...')

    # Find all image paths for class
    names = []
    for imfile in os.listdir(folder):
        if imfile.endswith('.jpg'):
            names.append(imfile)

    # Randomly split 80-20 
    train,val = train_test_split(names,test_size=0.2)

    # Create validation folder
    save_dir = sys.argv[1]+'/val/'+item_type+'/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save split data to validation folder
    for item in val:
        os.rename(folder+item,save_dir+item)
