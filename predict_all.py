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


# Identify data directory and model to load
base_dir = sys.argv[1]
model_num = sys.argv[2]

model_name = 'model'+str(model_num)
model = load_model('models/'+model_name+'.h5')

# Set up results folder and files
results_dir = 'results/'+model_name+'/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

corr_file = results_dir+'correct.txt'
incorr_file = results_dir+'incorrect.txt'
perf_file = results_dir+'performance.txt'


def generator(data_type):       # data_type = 'train' or 'val' depending on data being tested

    class_acc = defaultdict(lambda: [0,0])

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

    # Loop through files and store correct and incorrect filenames separately
    correct = open(corr_file,'a')
    incorrect = open(incorr_file,'a')
    for i in range(steps):
        #print(filenames[i]+'\t'+predictions[i]+'\t'+str(pred_vals[i]))
        f = filenames[i].split('/')[0]
        if predictions[i] == f:
            class_acc[f][0] += 1
            correct.write(filenames[i]+'\n')
        else:
            class_acc[f][1] += 1
            incorrect.write(filenames[i]+'\n')

    # Save total and per-class statistics
    with open(perf_file,'a') as perf:
        perf.write(model_name+': '+data_type+'\n')
        count = 0
        rmse = 0
        correct = 0
        incorrect = 0
        # Accumulate stats from each class
        for i in class_acc:
            count += 1
            correct += class_acc[i][0] 
            incorrect += class_acc[i][1]
            acc = float(class_acc[i][0])/float(class_acc[i][0]+class_acc[i][1])

            # Accumulate RMSE values
            rmse += (float(1-acc))**2

            # Save class accuracy
            perf.write(i+': '+str(acc)+'\n')

        # Calculate and save total accuracy and RMSE
        rmse /= float(count)
        rmse = float(rmse) ** (0.5)
        perf.write('total: '+str(float(correct)/float(correct+incorrect))+'\n')
        perf.write('rmse: '+str(rmse))
        
        perf.write('\n\n')



# Clear contents of save files
a = open(corr_file,'w+')
a.close()
b = open(incorr_file,'w+')
b.close()
c = open(perf_file,'w+')
c.close()

# Run generator predictions on training and validation data
generator('train')
generator('val')
