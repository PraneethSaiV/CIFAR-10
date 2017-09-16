import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.preprocessing import OneHotEncoder
os.chdir('D:\Cifar-10')

# Data loading function
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_batch(file):
	data_batch = unpickle(file)
	data_array = data_batch[b'data']
	data_labels = np.array(data_batch[b'labels']).reshape([-1,1])
	encode = OneHotEncoder(sparse = False)
	return data_array, np.array(encode.fit_transform(data_labels))
#np.transpose(data_array.reshape([1000,3,32,32]),[0,2,3,1])

def next_batch(data_array, data_labels, k):
	#data_array = np.array(data_array
	#data_labels = np.array(data_labels)
	max_number = data_labels.shape[0]
	indexes = random.sample([i for i in range(max_number)],k)
	return data_array[indexes,:], data_labels[indexes,:]

#x_array = data_batch_1['data'].reshape([10000,3,32,32])
x = tf.placeholder(tf.float32, shape = [None,3072])
y_ = tf.placeholder(tf.float32, shape = [None,10])

# Initialization functions
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape = shape)
	return tf.Variable(initial)

def conv2d(x , W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2d(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

# Data
x_initial = tf.transpose(tf.reshape(x,[-1,3,32,32]), perm = [0,2,3,1])

# Conv Layer 1
W1_conv = weight_variable([5,5,3,32])   # We choose 32 outputs to this layer
b1_conv = bias_variable([32])	# We chose 32 above
h_conv1 = tf.nn.relu(conv2d(x_initial, W1_conv) + b1_conv)
h_pool1 = max_pool_2d(h_conv1)

# Conv Layer 2
W2_conv = weight_variable([5,5,32,64])
b2_conv = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W2_conv) + b2_conv)
h_pool2 = max_pool_2d(h_conv2)

# Fully Connected 
W1_full = weight_variable([8*8*64,1024])
b1_full = bias_variable([1024])
reshaped = tf.reshape(h_pool2,[-1,8*8*64])
h_full1 = tf.nn.relu(tf.matmul(reshaped, W1_full) + b1_full)

# Dropout
P_keep = tf.placeholder(tf.float32)
h_full1_dropout = tf.nn.dropout(h_full1,P_keep)

# Fully Connected
W2_full = weight_variable([1024,10])
b2_full = bias_variable([10])
y_predicted = tf.matmul(h_full1_dropout,W2_full) + b2_full

# Loss function and other parameters
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_predicted))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
is_correct = tf.equal(tf.argmax(y_predicted,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Training
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#for batch in enumerate(['data_batch_1']):
	x, y = get_batch('data_batch_1')
	for i in range(1000):
		x_batch, y_batch = next_batch(x,y,50)
		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict = {x: x_batch, y_: y_batch, P_keep: 1.0})
			print('For ',i,'training accuracy', train_accuracy)
		train_step.run(feed_dict = {x: x_batch,y: y_batch, P_keep: 0.5})


