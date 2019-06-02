#!/usr/bin/env python

import os
import sys
from PIL import Image
from glob import glob

# Loop through each class in specified directory
for folder in glob(sys.argv[1] + '/*/'):
    count = 0
    print(folder)
    for imfile in os.listdir(folder):
        if imfile.endswith('.jpg'):
            # Try opening image, delete if fails
            try:
                im = Image.open(folder + imfile)
                im.verify()
            except:
                os.remove(folder + imfile)
                count += 1
    # Print number of removed images
    print('\t' + str(count))
