# Python program to implement a 
# single neuron neural network 

# import all necessery libraries 
from numpy import exp, array, random, dot, tanh
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 


import pandas as pd
 

# Class to create a neural 
# network with single neuron 
class NeuralNetwork(): 
    
    def __init__(self): 
        
        # Using seed to make sure it'll 
        # generate same weights in every run 
        random.seed(1) 
        
        # 3x1 Weight matrix 
        self.weight_matrix = 2 * random.random((5, 1)) - 1
        print('init: weight matrix')
        print(self.weight_matrix)
    # tanh as activation function 
    def tanh(self, x): 
        print('inside tanh')
        return tanh(x) 

    # derivative of tanh function. 
    # Needed to calculate the gradients. 
    def tanh_derivative(self, x): 
        print('inside tanh_derivative')
        return 1.0 - tanh(x) ** 2

    # forward propagation 
    def forward_propagation(self, inputs):
        print('forward_propagation: weight matrix')
        print(self.weight_matrix)
        print('inputs')
        print(inputs)
        return self.tanh(dot(inputs, self.weight_matrix)) 
    
    # training the neural network. 
    def train(self, train_inputs, train_outputs, 
                            num_train_iterations): 
                                
        # Number of iterations we want to 
        # perform for this set of input. 
        print('train:train inputs')
        print(train_inputs)
        print('train outputs')
        print(train_outputs)
        print(num_train_iterations)
        for iteration in range(num_train_iterations): 
            print('loop')
            print(iteration)
            #iteration for number of rows
            for record in train_inputs:
                output = self.forward_propagation(record) 
                print('calculated_output')
                print(output)
                # Calculate the error in the output. 
                error = train_outputs - output 
                print('error')
                print(error)
                # multiply the error by input and then 
                # by gradient of tanh funtion to calculate 
                # the adjustment needs to be made in weights 
                adjustment = dot(train_inputs.T, error *
                            self.tanh_derivative(output)) 
                print('adjustment')
                print(adjustment)                        
                # Adjust the weight matrix 
                self.weight_matrix += adjustment 

# Driver Code 
if __name__ == "__main__": 
    
    neural_network = NeuralNetwork() 
    
    print ('main:Random weights at the start of training') 
    print (neural_network.weight_matrix) 

    #train_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]) 
    #train_outputs = array([[0, 1, 1, 0]]).T 
    df = pd.read_csv ('E:\\GoogleDrive_sem1\\GoogleDrive_sem1\\sem 4\\project\\2019201015\\data\\combined_csv.csv')
    print(df)

    #work = df[["Date","Prev Close","Open Price","High Price","Low Price","Close Price"]]
    work = df[["Date","Prev Close","Open Price","High Price","Low Price","Close Price"]]

    work['O-C'] = work['Open Price'] - work['Close Price']
    work['H-L'] = work['High Price'] - work['Low Price']
    
    
    train_inputs=work[["Open Price","High Price","Low Price","O-C","H-L"]]
    train_inputs=preprocessing.normalize(train_inputs)

    #train_inputs['Open Price']=preprocessing.normalize(train_inputs['Open Price'])
    #train_inputs['High Price']=preprocessing.normalize(train_inputs['High Price'])
    #train_inputs['Low Price']=preprocessing.normalize(train_inputs['Low Price'])
    #train_inputs['O-C']=preprocessing.normalize(train_inputs['O-C'])
    #train_inputs['H-L']=preprocessing.normalize(train_inputs['H-L'])

    train_outputs = work['Close Price']
    #train_outputs=preprocessing.normalize(train_outputs)
    train_outputs = MinMaxScaler().fit_transform(array(train_outputs).reshape(-1,1)) 

    neural_network.train(train_inputs, train_outputs, 10) 

    print ('New weights after training') 
    print (neural_network.weight_matrix) 

    # Test the neural network with a new situation. 
    print ("Testing network on new examples ->") 
    testRecord = MinMaxScaler().fit_transform(array([2168,2183.9,2154,29.9,0.4]).reshape(-1,1))
    print (neural_network.forward_propagation(testRecord.reshape(1,5))) 
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

