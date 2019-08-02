#!/usr/bin/env python


#########################################################
# This code tests a model on all training and validation
# data. It prints the files it classifies correctly to
# 'correct.txt' and the ones it classifies incorrectly
# to 'incorrect.txt'.
#
# Usage: ./predict_all.py data/ x
#       (x is the model number)
#########################################################


from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os,sys
from collections import defaultdict


corr_file = 'correct.txt'
incorr_file = 'incorrect.txt'

# Identify data directory and model to load
base_dir = sys.argv[1]
model_num = sys.argv[2]

model_name = 'model'+str(model_num)
model = load_model('models/'+model_name+'.h5')

def generator(data_type):       # data_type = 'train' or 'val' depending on data being tested
    # Identify data directory
    data_dir = os.path.join(base_dir, data_type)
    data_datagen = ImageDataGenerator(rescale=1./255)

    # Initialize data generator
    data_generator = data_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode='categorical')

    filenames=data_generator.filenames

    # Get class labels from generator
    labels = (data_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    # Make model predictions and extract best pred
    data_generator.reset()
    steps=len(filenames)
    pred = model.predict_generator(data_generator,steps=steps,verbose=1)
    predicted_class_indices=np.argmax(pred,axis=1)
    predictions = [labels[i] for i in predicted_class_indices]
    pred_vals = np.max(pred,axis=1)

    # Loop through files and store file name according to correctness
    correct = open(corr_file,'a+')
    incorrect = open(incorr_file,'a+')
    for i in range(steps):
        print(filenames[i]+'\t'+predictions[i]+'\t'+str(pred_vals[i]))
        f = filenames[i].split('/')[0]
        if predictions[i] == f:
            correct.write(filenames[i]+'\n')
        else:
            incorrect.write(filenames[i]+'\n')

# Clear contents
a = open(corr_file,'w+')
a.close()
b = open(incorr_file,'w+')
b.close()

# Run generator predictions on training and validation data
generator('val')
generator('train')
