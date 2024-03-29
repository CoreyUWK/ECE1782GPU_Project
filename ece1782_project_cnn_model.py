# -*- coding: utf-8 -*-
"""ECE1782 Project CNN Model

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hYGc6HYKjkdfxS8GRcrTaZxHtsO-2E3R

# Convolutional Neural Network (CNN)

### Import TensorFlow
"""

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import time
import pprint
!nvidia-smi

"""Paper CNN Tensorflow Model"""

# Setup input as all ones
input_shape = (1, 1, 56, 100) # Batch Size, Input Channels, Rows, Columns
x = tf.ones(input_shape)

# Setup model
model = models.Sequential()

# Convolution Layer 1: 1 Input, 64 Output, Same dimensions
model.add(layers.Conv2D(64, (8, 8), activation='relu', strides=1, padding='same', input_shape=input_shape[1:], kernel_initializer="ones", bias_initializer="ones", data_format='channels_first'))
#cov1 = model(x)
#print("cov1:", cov1.shape)

# Maxpool 1: 
model.add(layers.MaxPooling2D((2, 2), strides=2, padding='same', data_format='channels_first'))
#maxpool1 = model(x)
#print("maxpool1:", maxpool1.shape)

# Convolution Layer 2: 64 Input, 64 Output, Same dimensions
model.add(layers.Conv2D(64, (4, 4), activation='relu', strides=1, padding='same', kernel_initializer="ones", bias_initializer="ones", data_format='channels_first'))
#cov2 = model(x)
#print("cov2:", cov2.shape)

# Maxpool 2: 
model.add(layers.MaxPooling2D((2, 2), strides=2, padding='same', data_format='channels_first'))
#maxpool2 = model(x)
#print("maxpool2:", maxpool2.shape)

# Convolution Layer 3: 64 Input, 64 Output, Same dimensions
model.add(layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding='same', kernel_initializer="ones", bias_initializer="ones", data_format='channels_first'))
#cov3 = model(x)
#print("cov3:", cov3.shape)

# Fully Connection Layers
##model.add(layers.GlobalAveragePooling2D()) # Not use
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', kernel_initializer="ones", bias_initializer="ones"))
model.add(layers.Dense(3, activation='softmax', kernel_initializer="ones", bias_initializer="ones"))
##model.add(layers.Softmax()) # Not use
#model.summary()

# Get timing results
runs = 10
for i in range(0, runs):
  start_time = time.time()
  y = model(x)
  print("--- %s ms for 1 ---" % ((time.time() - start_time)*1000))

# To save to file 
# https://colab.research.google.com/github/d2l-ai/d2l-en-colab/blob/master/chapter_deep-learning-computation/read-write.ipynb

# Print Results
print("Input:")
print(x.shape)
print(x)

print("Output:")
print(y.shape)
#print(tf.shape(y))
tf.print(y, summarize=5)

"""Get Model Timing Results when Batching instead of looping"""

# Setup input as all ones
input_shape = (100, 1, 56, 100) # Batch Size, Input Channels, Rows, Columns
x = tf.ones(input_shape)

# Setup model
model_batch = models.Sequential()

# Convolution Layer 1: 1 Input, 64 Output, Same dimensions
model_batch.add(layers.Conv2D(64, (8, 8), activation='relu', strides=1, padding='same', input_shape=input_shape[1:], kernel_initializer="ones", bias_initializer="ones", data_format='channels_first'))
#cov1 = model_batch(x)
#print("cov1:", cov1.shape)

# Maxpool 1: 
model_batch.add(layers.MaxPooling2D((2, 2), strides=2, padding='same', data_format='channels_first'))
#maxpool1 = model_batch(x)
#print("maxpool1:", maxpool1.shape)

# Convolution Layer 2: 64 Input, 64 Output, Same dimensions
model_batch.add(layers.Conv2D(64, (4, 4), activation='relu', strides=1, padding='same', kernel_initializer="ones", bias_initializer="ones", data_format='channels_first'))
#cov2 = model_batch(x)
#print("cov2:", cov2.shape)

# Maxpool 2: 
model_batch.add(layers.MaxPooling2D((2, 2), strides=2, padding='same', data_format='channels_first'))
#maxpool2 = model_batch(x)
#print("maxpool2:", maxpool2.shape)

# Convolution Layer 3: 64 Input, 64 Output, Same dimensions
model_batch.add(layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding='same', kernel_initializer="ones", bias_initializer="ones", data_format='channels_first'))
#cov3 = model_batch(x)
#print("cov3:", cov3.shape)

# Fully Connection Layers
##model_batch.add(layers.GlobalAveragePooling2D()) # Not use
model_batch.add(layers.Flatten())
model_batch.add(layers.Dense(256, activation='relu', kernel_initializer="ones", bias_initializer="ones"))
model_batch.add(layers.Dense(3, activation='softmax', kernel_initializer="ones", bias_initializer="ones"))
##model_batch.add(layers.Softmax()) # Not use
#model_batch.summary()

# Get timing results
runs = 10
for i in range(0, runs):
  start_time = time.time()
  y = model_batch(x)
  print("--- %s ms for 1 ---" % ((time.time() - start_time)*1000))

# To save to file 
# https://colab.research.google.com/github/d2l-ai/d2l-en-colab/blob/master/chapter_deep-learning-computation/read-write.ipynb

# Print Results
print("Input:")
print(x.shape)
print(x)

print("Output:")
print(y.shape)
#print(tf.shape(y))
tf.print(y, summarize=5)

"""Try Model but with MaxPool Stride=1"""

# Setup input as all ones
input_shape = (1, 1, 56, 100) # Batch Size, Input Channels, Rows, Columns
x = tf.ones(input_shape)

# Setup model
model_maxpool_stride1 = models.Sequential()

# Convolution Layer 1: 1 Input, 64 Output, Same dimensions
model_maxpool_stride1.add(layers.Conv2D(64, (8, 8), activation='relu', strides=1, padding='same', input_shape=input_shape[1:], kernel_initializer="ones", bias_initializer="ones", data_format='channels_first'))
#cov1 = model_maxpool_stride1(x)
#print("cov1:", cov1.shape)

# Maxpool 1: 
model_maxpool_stride1.add(layers.MaxPooling2D((2, 2), strides=1, padding='same', data_format='channels_first'))
#maxpool1 = model_maxpool_stride1(x)
#print("maxpool1:", maxpool1.shape)

# Convolution Layer 2: 64 Input, 64 Output, Same dimensions
model_maxpool_stride1.add(layers.Conv2D(64, (4, 4), activation='relu', strides=1, padding='same', kernel_initializer="ones", bias_initializer="ones", data_format='channels_first'))
#cov2 = model_maxpool_stride1(x)
#print("cov2:", cov2.shape)

# Maxpool 2: 
model_maxpool_stride1.add(layers.MaxPooling2D((2, 2), strides=1, padding='same', data_format='channels_first'))
#maxpool2 = model_maxpool_stride1(x)
#print("maxpool2:", maxpool2.shape)

# Convolution Layer 3: 64 Input, 64 Output, Same dimensions
model_maxpool_stride1.add(layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding='same', kernel_initializer="ones", bias_initializer="ones", data_format='channels_first'))
#cov3 = model_maxpool_stride1(x)
#print("cov3:", cov3.shape)

# Fully Connection Layers
##model_maxpool_stride1.add(layers.GlobalAveragePooling2D()) # Not use
model_maxpool_stride1.add(layers.Flatten())
model_maxpool_stride1.add(layers.Dense(256, activation='relu', kernel_initializer="ones", bias_initializer="ones"))
model_maxpool_stride1.add(layers.Dense(3, activation='softmax', kernel_initializer="ones", bias_initializer="ones"))
##model_maxpool_stride1.add(layers.Softmax()) # Not use
#model_maxpool_stride1.summary()

# Get timing results
runs = 10
for i in range(0, runs):
  start_time = time.time()
  y = model_maxpool_stride1(x)
  print("--- %s ms for 1 ---" % ((time.time() - start_time)*1000))

# To save to file 
# https://colab.research.google.com/github/d2l-ai/d2l-en-colab/blob/master/chapter_deep-learning-computation/read-write.ipynb

# Print Results
print("Input:")
print(x.shape)
print(x)

print("Output:")
print(y.shape)
#print(tf.shape(y))
tf.print(y, summarize=5)

"""Try with random weight values to see if the fast times from above tests is due to some caching"""

# Setup input as all ones
input_shape = (1, 1, 56, 100) # Batch Size, Input Channels, Rows, Columns
x = tf.ones(input_shape)

init_val = "RandomUniform"

# Setup model
model_rand = models.Sequential()

# Convolution Layer 1: 1 Input, 64 Output, Same dimensions
model_rand.add(layers.Conv2D(64, (8, 8), activation='relu', strides=1, padding='same', input_shape=input_shape[1:], kernel_initializer=init_val, bias_initializer=init_val, data_format='channels_first'))
#cov1 = model_rand(x)
#print("cov1:", cov1.shape)

# Maxpool 1: 
model_rand.add(layers.MaxPooling2D((2, 2), strides=1, padding='same', data_format='channels_first'))
#maxpool1 = model_rand(x)
#print("maxpool1:", maxpool1.shape)

# Convolution Layer 2: 64 Input, 64 Output, Same dimensions
model_rand.add(layers.Conv2D(64, (4, 4), activation='relu', strides=1, padding='same', kernel_initializer=init_val, bias_initializer=init_val, data_format='channels_first'))
#cov2 = model_rand(x)
#print("cov2:", cov2.shape)

# Maxpool 2: 
model_rand.add(layers.MaxPooling2D((2, 2), strides=1, padding='same', data_format='channels_first'))
#maxpool2 = model_rand(x)
#print("maxpool2:", maxpool2.shape)

# Convolution Layer 3: 64 Input, 64 Output, Same dimensions
model_rand.add(layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding='same', kernel_initializer=init_val, bias_initializer=init_val, data_format='channels_first'))
#cov3 = model_rand(x)
#print("cov3:", cov3.shape)

# Fully Connection Layers
##model_rand.add(layers.GlobalAveragePooling2D()) # Not use
model_rand.add(layers.Flatten())
model_rand.add(layers.Dense(256, activation='relu', kernel_initializer=init_val, bias_initializer=init_val))
model_rand.add(layers.Dense(3, activation='softmax', kernel_initializer=init_val, bias_initializer=init_val))
##model_rand.add(layers.Softmax()) # Not use
#model_rand.summary()

# Get timing results
runs = 10
for i in range(0, runs):
  start_time = time.time()
  y = model_rand(x)
  print("--- %s ms for 1 ---" % ((time.time() - start_time)*1000))

# To save to file 
# https://colab.research.google.com/github/d2l-ai/d2l-en-colab/blob/master/chapter_deep-learning-computation/read-write.ipynb

# Print Results
print("Input:")
print(x.shape)
print(x)

print("Output:")
print(y.shape)
#print(tf.shape(y))
tf.print(y, summarize=5)

"""Now Try with objax"""

#Install Objax
!pip --quiet install  objax
import objax
import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import jax.numpy as jn
import random 
import matplotlib.pyplot as plt
from pprint import  pprint
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

X_train = train_images.transpose(0, 3, 1, 2) / 255.0
Y_train = train_labels.flatten()
X_test = test_images.transpose(0, 3, 1, 2) / 255.0
Y_test = test_labels.flatten()

class ConvNet(objax.Module):
  def __init__(self, number_of_channels = 3, number_of_classes = 10):
    self.conv_1 = objax.nn.Sequential([objax.nn.Conv2D(number_of_channels, 32, 3, padding="VALID"), objax.functional.relu])
    self.conv_2 = objax.nn.Sequential([objax.nn.Conv2D(32, 64, 3, padding="VALID"), objax.functional.relu])
    self.conv_3 = objax.nn.Sequential([objax.nn.Conv2D(64, 64, 3, padding="VALID"), objax.functional.relu])

    self.linear1 = objax.nn.Sequential([objax.nn.Linear(1024, 64), objax.functional.relu])
    self.linear2 = objax.nn.Linear(64, number_of_classes)

  def __call__(self, x):
    #print(x.shape)
    x = objax.functional.max_pool_2d(self.conv_1(x), 2)
    #print(x.shape)
    x = objax.functional.max_pool_2d(self.conv_2(x), 2)
    #print(x.shape)
    x = self.conv_3(x)
    #print(x.shape)
    x = objax.functional.flatten(x)
    #print(x.shape)
    x = self.linear1(x)
    #print(x.shape)
    x = self.linear2(x)
    #print(x.shape)
    return x

#The following line creates the CNN
model = ConvNet()
#You can examine the architecture of our CNN by calling model.vars()
print(model.vars())

#Define loss function as averaged value of of cross entropies
def loss_function(x, labels):
    logit = model(x)
    return objax.functional.loss.cross_entropy_logits_sparse(logit, labels).mean()

#Define a prediction function
predict = objax.Jit(lambda x: objax.functional.softmax(model(x)), model.vars()) 

#Create an object that can be used to calculate the gradient and value of loss_function
gv= objax.GradValues(loss_function, model.vars())

#Create an object that can be used to provide trainable variables in the model
tv = objax.ModuleList(objax.TrainRef(x) for x in model.vars().subset(objax.TrainVar))

opt = objax.optimizer.Adam(model.vars())

#Training routine
def train_op(x, y, learning_rate):
    lr = learning_rate
    gradient, loss_value = gv(x, y)   # calculate gradient and loss value "backprop"
    for grad, params in zip(gradient, tv.vars()):
      params.value -= lr * grad
    #opt(lr, gradient)  # update weights
    return loss_value                      # return loss value

#make train_op (much) faster using JIT compilation
train_op = objax.Jit(train_op, gv.vars() + tv.vars())

def train(EPOCHS = 10, BATCH = 32, LEARNING_RATE = 9e-4):
  avg_train_loss_epoch = []
  train_acc_epoch = []

  for epoch in range(EPOCHS):
      avg_train_loss = 0 # (averaged) training loss per batch
      train_acc = 0      # training accuracy per batch

      # shuffle the examples prior to training to remove correlation 
      train_indices = np.arange(len(X_train)) 
      #np.random.shuffle(train_indices)
      for it in range(0, X_train.shape[0], BATCH):
          #print("{}:{}".format(it, it + BATCH))
          batch = train_indices[it : it + BATCH] #PUT YOUR CODE HERE#
          #print(X_train[batch])
          #print(len(batch))
          avg_train_loss += float(train_op(X_train[batch], Y_train[batch], LEARNING_RATE)[0]) * len(batch)
          train_prediction = predict(X_train[batch]).argmax(1)
          train_acc += (np.array(train_prediction).flatten() == Y_train[batch]).sum()
      train_acc_epoch.append(train_acc/X_train.shape[0])
      avg_train_loss_epoch.append(avg_train_loss/X_train.shape[0])
      print('Epoch %04d Training Loss %.2f Training Accuracy %.2f' % (epoch + 1, avg_train_loss/X_train.shape[0], 100*train_acc/X_train.shape[0]))

  #Plot training loss
  plt.title("Train Loss")
  plt.plot(avg_train_loss_epoch, label="Train")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()

  plt.title("Train")
  plt.plot(train_acc_epoch, label="Train")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.legend(loc='best')
  plt.show()

def test(BATCH = 32):
  avg_test_loss = 0 # (averaged) test loss per batch
  test_acc = 0      # test accuracy per batch

  # run test set
  test_indices = np.arange(len(X_test)) 
  #np.random.shuffle(test_indices)    
  for it in range(0, X_test.shape[0], BATCH):
      batch = test_indices[it : it + BATCH] #PUT YOUR CODE HERE#
      avg_test_loss += float(loss_function(X_test[batch], Y_test[batch])) * len(batch)
      test_prediction = predict(X_test[batch]).argmax(1)
      test_acc += (np.array(test_prediction).flatten() == Y_test[batch]).sum()
  test_acc = (test_acc/X_test.shape[0])
  avg_test_loss = (avg_test_loss/X_test.shape[0])

  print('Test Loss %.2f Test Accuracy %.2f' % (avg_test_loss, 100*test_acc))

train()

test()

"""Extra Debug Stuff - Can Ignore"""

# With `padding` as "same".
input_shape = (1, 4, 5, 1)
x = tf.ones(input_shape)
model = tf.keras.layers.Conv2D(1, 2, padding="same", input_shape=input_shape, kernel_initializer="ones", bias_initializer="zeros")
y = model(x)

print("Input:")
pprint.pprint(x)
print("Output:")
pprint.pprint(y)

print("Weights:")
print(model.get_config(), model.get_weights())
weights = model.get_weights()
pprint.pprint(weights)

#input = tf.ones((5, 4), dtype=tf.float32)
#kernel = tf.ones((2, 2), dtype=tf.float32)
#print(input)
#conv = tf.nn.conv2d(input, kernel, 1, 'SAME')
#print(conv)

# With `padding` as "same".
input_shape = (1, 56, 100, 1)
x = tf.ones(input_shape)
model = tf.keras.layers.Conv2D(1, 2, padding="same", input_shape=input_shape[1:], kernel_initializer="ones", bias_initializer="zeros")
y = model(x)

print("Input:")
pprint.pprint(x)
print("Output:")
pprint.pprint(y)

print("Weights:")
print(model.get_config(), model.get_weights())
weights = model.get_weights()
pprint.pprint(weights)

"""When starting with new runtime: 
27.790482997894287 seconds for 1

Second time running:
0.00956869125366211 seconds

Third time:
0.005471467971801758 seconds

0.009866952896118164 seconds 
0.00510859489440918 seconds
"""

input_shape = (1, 1, 4, 2)
x = tf.ones(input_shape)
print(x)