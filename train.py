#!/usr/bin/env python

import os
import sys
import keras
import numpy as np
from keras.applications import mobilenet
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Dropout
from keras.optimizers import RMSprop
from keras.models import Model
from keras import models
from keras.preprocessing.image import ImageDataGenerator




def save_hyperparams(drop,batch,opt,extra_layers,layer_size,act,patience,init_weights,model_name,train_items):
    with open('models.txt','a') as f:
        f.write(model_name+'\n')
        f.write('Dropout ratio: '+str(drop)+'\n')
        f.write('Batch size: '+str(batch)+'\n')
        f.write('Optimizer: '+opt+'\n')
        f.write('Hidden Layer: '+str(extra_layers)+'\n')
        f.write('Hidden Layer Size: '+str(layer_size)+'\n')
        f.write('Hidden Layer Activation: '+act+'\n')
        f.write('Patience: '+str(patience)+'\n')
        f.write('Preloaded weights: '+init_weights+'\n')
        f.write('Classes: '+str(train_items)+'\n\n')


# Select hyperparameters
drop = 0.4
batch = 16
opt = 'adam'
extra_layers = 1
if extra_layers:
    layer_size = 256
    act = 'relu'
else:
    layer_size = 0
    act = 'none'
patience = 5
init_weights = 'imagenet'

# Pick model name
model_dir = 'models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_list = os.listdir(model_dir)
model_id = 0
while True:
    model_name = 'model' + str(model_id) + '.h5'
    if model_name in model_list:
        model_id += 1
    else:
        break
model_file = 'models/' + model_name
model_name = model_name[:-3]

print("Saving model to: " + model_file)

# Identify data directory
base_dir = sys.argv[1]
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Confirm train and val contain identical classes
train_items = set(os.listdir(train_dir))
val_items = set(os.listdir(val_dir))
if train_items != val_items:
    print('Training classes not same as validation classes')
    print('\tItems in train not in val: ' + str((train_items - val_items)))
    print('\tItems in val not in train: ' + str((val_items - train_items)))
    quit()
num_classes = len(train_items)

# Store hyperparameter settings
save_hyperparams(drop,batch,opt,extra_layers,layer_size,act,patience,init_weights,model_name,train_items)

# Initiate data generators
train_datagen = ImageDataGenerator(
    rescale=1./255)
val_datagen = ImageDataGenerator(
    rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch,
    class_mode='categorical')
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical')

# Build model architecture from MobileNet
mbn = mobilenet.MobileNet(weights=init_weights,include_top=False,input_shape=(224,224,3))
for layer in mbn.layers:
    layer.trainable = False
model = models.Sequential()
model.add(mbn)
model.add(GlobalAveragePooling2D())
if extra_layers:
    model.add(Dense(layer_size,activation=act))
model.add(Dropout(drop))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callback functions
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
model_save = ModelCheckpoint(model_file, monitor='val_loss', mode='min', save_best_only=True)

# Train model
steps = train_generator.n//train_generator.batch_size
history = model.fit_generator(
      train_generator,
      validation_data=val_generator,
      steps_per_epoch=steps,
      epochs=100,
      validation_steps=val_generator.n,
      callbacks=[early_stop,model_save],
      verbose=1)

# Save training output
out_dir = 'outputs/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
with open(out_dir+model_name+'.txt','w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write('\n')
    h = history.history
    #f.write('acc\t\t\tloss\t\t\t\tval_acc\t\tval_loss\n')
    for i in range(len(h['acc'])):
        #f.write(str(history.history))
        #f.write(str(h['acc'][i])+'\t\t'+str(h['loss'][i])+'\t\t'+str(h['val_acc'][i])+'\t\t'+str(h['val_loss'][i])+'\n')
        f.write('Epoch %d\n' % i)
        f.write('loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f\n' % (h['loss'][i],h['acc'][i],h['val_loss'][i],h['val_acc'][i]))



