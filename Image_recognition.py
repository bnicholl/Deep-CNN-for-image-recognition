#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:31:44 2017
@author: bennicholl
"""
"""IMPORTANT!!!  if you want to run algorithm, you must go down to line 204 and 210 and change the path
   in order to save the weights and tnesorboard"""

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

np.set_printoptions(threshold=np.nan)


"""in order to use the array of pixel values we must put the array into a
four dimensional array in order for tensorflow to be able to use the array"""
"""here we will be using the mnist dataset, and reshaping them as neccesary"""

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x_trains = mnist.train.images
x_trains = np.reshape(x_trains,[-1,28,28,1])
y_trains = mnist.train.labels

x_test = mnist.test.images
x_test = np.reshape(x_test,[-1,28,28,1])
y_test = mnist.test.labels

np.set_printoptions(threshold=np.nan)

"""the list below is  [batch, height, width, channels]"""
#images = np.reshape(images,(-1,110,110,1))

"""create an instance of the class with two required arguments, image and labels"""
"""images are the training images to be used. labels are the images respective labels"""
class convolution():
    # ensure input_var is in 4 dimensions
    def __init__(self, images, labels):
        
        self.images = images
        self.labels = labels
        
        """these are the placeholders for our two input arguments directly above"""
        self.inputs = tf.placeholder(tf.float32, name = 'inputs')
        self.outputs = tf.placeholder(tf.float32, name = 'outputs')
        
        self.reuse = None        
        self.conv1()       
            
    """our first convolution of our images"""    
    def conv1(self, padding = 'VALID'):
           
        with tf.variable_scope('convolutions'):
            self.filters = tf.get_variable("filters", dtype=tf.float32, 
            initializer=tf.random_normal([4,4,1,2], mean = 0.3, stddev=0.5))
            
            self.data = tf.nn.convolution(self.inputs, self.filters, padding)           
            """makes all negative numbers 0"""
            self.data = tf.nn.relu(self.data) 
            
            self.pool1([1,3,2,1], [1,3,2,1])    
            
    """1st pool layer. Value is our output of conv1, kernel_size is size of,
    our window to average strides is how many neurons(pixels) our kernel will move"""
    def pool1(self, kernal_size, strides, padding = 'SAME'):
        
        self.data = tf.nn.avg_pool(self.data, kernal_size, strides, padding)
               
        self.conv2()   
            
    """second convolution"""    
    def conv2(self, padding = 'VALID'):
        self.filters2 = tf.get_variable("filters2", dtype=tf.float32, 
        initializer=tf.random_normal([4,4,2,1], mean = 0.5, stddev=0.5)) 
            
        self.data = tf.nn.convolution(self.data, self.filters2, padding)
        self.data = tf.nn.relu(self.data)
        
        self.pool2([1,2,2,1],[1,2,2,1])
    
    """second pool layer"""
    def pool2(self, kernal_size, strides, padding = 'SAME'):

        self.data = tf.nn.avg_pool(self.data, kernal_size, strides, padding)
               
        self.flatten()      
    
    """flattens our array of pixels into a vector, this puts our data in standard "features
    form", and enables matrix multiplication linear transformation functions on our features/weights"""    
    def flatten(self):
        self.data = tf.contrib.layers.flatten(self.data)
        self.fully_connected()
      
    def fully_connected(self):
        with tf.variable_scope('hidden_layers'):
            """create a matrix of weights where Y axis(first int) is == to the shape of our
            X axis, and X axis(second int) produces the amount of output nodes we want"""
      
            """we need to create a session here so we can save the 1st index of our data, and
            plug it into our self.weights variable below to ensure propper matrix multiplication"""
            with tf.Session() as session:
                session.run(self.filters.initializer)
                session.run(self.filters2.initializer)
                feed_dict = {self.inputs : self.images, self.outputs : self.labels}    
                shapes = session.run(self.data, feed_dict)
                shapes = shapes.shape[1]
            
            """create first set of weights to be matrix mutlipled by features"""
            self.weights = tf.get_variable("weights", shape=[shapes, 14],
            initializer=tf.contrib.layers.xavier_initializer())                             
            
            """create first bias to be added to matrix multipliation problem"""
            self.bias = tf.get_variable('bias',dtype = tf.float32, 
            initializer = tf.random_normal([1]))
            
            """first hidden layer is calculated. equation is simple linear MX + B"""
            self.hidden_layer = tf.matmul(self.data, self.weights) + self.bias
            """non linear RELU activation function"""
            self.data = tf.nn.relu(self.hidden_layer)
            
            self.fully_connected2()
            
    def fully_connected2(self):
        
        """This does the same exact equation with different weights/updated variable as 
        the previous fully_connected function"""
        self.weights2 = tf.get_variable("weights2", shape=[14, 12],
        initializer=tf.contrib.layers.xavier_initializer())  
        self.bias2 = tf.get_variable('bias2',dtype = tf.float32, 
        initializer = tf.random_normal([1]))
        self.hidden_layer2 = tf.matmul(self.data, self.weights2, name = 'hidden_layer2') + self.bias2
        
        self.data = tf.nn.relu(self.hidden_layer2)
    
        self.fully_connected3()
        
    def fully_connected3(self):
        """This does the same exact equation with different weights/updated variable as 
        the previous fully_connected function"""
        self.weights3 = tf.get_variable("weights3", shape=[12, 10],
        initializer=tf.contrib.layers.xavier_initializer())  
 
        self.bias3 = tf.get_variable('bias3', dtype = tf.float32, 
        initializer = tf.random_normal([1]))    
        self.hidden_layer3 = tf.matmul(self.data, self.weights3) + self.bias3
        
        self.initialize_and_train()
        
                   
    def initialize_and_train(self):
        """this method is specificaly for testing phase. code below this is for training"""
        self.probabilities = tf.nn.softmax(self.hidden_layer3,name = 'test_probabilities')
        
        """Calulates 10 probabilities based off of our input nodes, than calculates the error using
        cross entropy function, which turns those ten probabilities into an integer value. we than take 
        the mean of the cross entropy errors. Logits are the values to be used as input to softmax"""
        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits = self.hidden_layer3, labels = self.outputs, name = 'error'))
        """initialize all of our variables with acutal numbers"""
        with tf.Session() as session:
            session.run(self.filters.initializer)
            session.run(self.filters2.initializer)
            session.run(self.weights.initializer)
            session.run(self.weights2.initializer)
            session.run(self.bias.initializer)
            session.run(self.bias2.initializer)
            session.run(self.weights3.initializer)
            session.run(self.bias3.initializer)
            """create gradient descent function"""
            self.train = tf.train.GradientDescentOptimizer(0.1).minimize(self.error)
            
            """these are our two index's that give us our batch size for gradient descent below"""
            index1 = 0
            index2 = 500
            """this for loop runs mini-batch gradient descent and prints error every ith iteration"""
            for i in range(4500): 
                """if our second index is less than the # of training sets, input propper index in feed_dict and run"""
                if index2 < int(self.images.shape[0]):                  
                    feed_dict = {self.inputs : self.images[index1:index2], self.outputs : self.labels[index1:index2]} 
                    session.run(self.train, feed_dict)
                    iteration = i+1
                    """add 500 to each index and continue iterations"""
                    index1 += 500
                    index2 += 500
                    
                elif index2 >= int(self.images.shape[0]):
                    """if our second index is greater than or equal to # of training sets, 
                    input propper index in feed_dict and run"""
                    index2 == int(self.images.shape[0])
                    feed_dict = {self.inputs : self.images[index1:index2], self.outputs : self.labels[index1:index2]}
                    session.run(self.train, feed_dict)
                    iteration = i+1
                    """reset the index back to its orginal value and continue iterations"""
                    index1 = 0
                    index2 = 500 

                if iteration % 100 == 0:  
                    print(index1,index2)
                    print('#', iteration, 'error is:', session.run(self.error, feed_dict))
            """save the final results of our weights/filter variables as outputfile"""
            self.saver = tf.train.Saver() 
            self.saver.save(session, "/Users/bennicholl/Desktop/outputfile")
            
            """this below code is for tensorboard, a data visualization tool"""
            """open local host:6006 on chrome, than type in hashtagged code block below in a terminal"""
            #python -m tensorboard.main --logdir="/Users/bennicholl/Desktop/output3"
            with tf.Session() as session:
                writer = tf.summary.FileWriter("/Users/bennicholl/Desktop/output3", session.graph)
                writer.close()  


# these are just examples of runnning x and y tests through one at a time. you can also 
# iterate thrugh the entire testset using x_test and y_test as arguments
x_ = x_test[80:96]
x_ = np.reshape(x_,[-1,28,28,1])
y_ = y_test[80:96]
y_ = y_.reshape(-1, 10)


"""run test data with this funtion after algo has been trained"""  
def restore(x,y): 
    with tf.Session() as ses:
        """import the graph that we saved after training"""
        saver = tf.train.import_meta_graph('outputfile.meta')
        saver.restore(ses,tf.train.latest_checkpoint('./'))
        
        """create instance of get_default_graph, which gets gets the specific 
        tensors in a graph"""
        graph = tf.get_default_graph()
        """get placeholders, than feed our test examples into our placeholders"""
        w1 = graph.get_tensor_by_name("inputs:0")
        w2 = graph.get_tensor_by_name('outputs:0')
        feed_dict = {w1 : x, w2 : y}        
        
        """restore self.error and self.probability function and then runs said functions"""
        restored_probability = graph.get_tensor_by_name('convolutions/hidden_layers/test_probabilities:0')       
        restored_error = graph.get_tensor_by_name('convolutions/hidden_layers/Mean:0')
        
        """printing val gives the probability of each element in a vector. only print for small batches"""
        val = ses.run(restored_probability,feed_dict)
        #print(ses.run(tf.as_string(val, scientific=None)))
        """this below code calculates the probability that are algorithm predicts the right image"""
        amount_correct = 0
        for e,i in enumerate(val):
            highest_prob = max(i)
            location_of_highest_prob = np.where(i == highest_prob)
            location_of_correct_label = np.where(y[e] ==1 )
            if location_of_highest_prob == location_of_correct_label:
                amount_correct +=1     
        probability_of_being_correct = amount_correct/len(y)
        print(probability_of_being_correct * 100, '% chance of correct prediciton')
        
        """gives you the acutal labels of each corresponding image"""
        #print(y_)
        """gives the cost function average."""
        print('cross entropy errors are:', ses.run(restored_error,feed_dict))
            

  
    
    
    
    

