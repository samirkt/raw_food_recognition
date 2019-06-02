#!/usr/bin/env python

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os,sys
from collections import defaultdict


class_acc = defaultdict(lambda: [0,0])

# Identify data directory and model to load
base_dir = sys.argv[1]
model_num = sys.argv[2]

model_name = 'model'+str(model_num)
model = load_model('models/'+model_name+'.h5')

# Identify validation directory
val_dir = os.path.join(base_dir, 'val')
val_datagen = ImageDataGenerator(rescale=1./255)

# Initialize validation data generator
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False,
    class_mode='categorical')

filenames=val_generator.filenames

# Get class labels from generator
labels = (val_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

# Make model predictions and extract best pred
val_generator.reset()
steps=len(filenames)
pred = model.predict_generator(val_generator,steps=steps,verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
predictions = [labels[i] for i in predicted_class_indices]
pred_vals = np.max(pred,axis=1)

# Accumulate per class prediction stats
for i in range(steps):
    #print(filenames[i]+'\t'+predictions[i]+'\t'+str(pred_vals[i]))
    f = filenames[i].split('/')[0]
    if predictions[i] == f:
        class_acc[f][0] += 1
    else:
        class_acc[f][1] += 1

# Save total and per-class statistics
with open('results.txt','a') as f:
    f.write(model_name+'\n')
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

        # Save class accuracty
        f.write(i+': '+str(acc)+'\n')

    # Calculate and save total accuracy and RMSE
    rmse /= float(count)
    rmse = float(rmse) ** (0.5)
    f.write('total: '+str(float(correct)/float(correct+incorrect))+'\n')
    f.write('rmse: '+str(rmse))
    
    f.write('\n\n')

