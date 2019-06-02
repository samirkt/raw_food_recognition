#!/usr/bin/env python

import os
import sys
from PIL import Image
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("error")

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(sys.argv[1],target_size=(224, 224),batch_size=1,shuffle=False)
filenames = generator.filenames
steps = len(filenames)
count = 0
generator.reset()
for i in range(steps):
    try:
        X,y = generator.next()
    except:
        os.remove(sys.argv[1] + filenames[i])
        count+=1
    sys.stdout.write('\r\t'+ str(i) + '/' + str(steps))
    sys.stdout.flush()
sys.stdout.write('\r\t'+str(steps)+'/'+str(steps)+'. Done.\n')
sys.stdout.write(str(count)+' images removed.\n')

quit()

'''
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
'''
