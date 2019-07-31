# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    # first, we compute distance matrix with x_train and test_datum, which is
    # the term ||x - x^(j)||^2
    dist_matrix = l2(test_datum.T, x_train)
    
    # we define partial denominator, the matrix shown in both numerator and 
    # denominator; then define numerator and denominator
    com_matrix = (-1) * (dist_matrix / (2.0 * (tau ** 2)))
    numerator = np.exp(com_matrix)
    denominator = np.exp(scipy.misc.logsumexp(com_matrix))
    
    # build matrix A
    a_values_vector = numerator / float(denominator)
    A = np.diag(a_values_vector[0 , : ])
    
    # compute w_star, optimal weight
    dimension = np.dot((np.dot(x_train.T, A)), x_train).shape[0]
    I_matrix = np.identity(dimension)
    coefficient = np.dot((np.dot(x_train.T, A)), x_train) + (lam * I_matrix)
    rhs = np.dot(np.dot(x_train.T, A), y_train)
    w_star = np.linalg.solve(coefficient, rhs)
    
    # predict y, call it y_hat, which is not a vector but a scaler since it is 
    # a predication for a single training example
    y_hat = np.dot(test_datum.T, w_star)
    
    return y_hat
    ## TODO

# helper function
def compute_losses(test_X, test_y, training_X, training_y, taus):
    losses = np.zeros(taus.shape[0])
    total = training_y.shape[0]
    y_hat_vector = np.zeros(test_X.shape[0])
    
    # compute loss for each tau
    for i in range(0, len(taus)):
        for j in range(0, test_X.shape[0]):
            single_loss = LRLS(test_X[j, : ].reshape(d,1), training_X, training_y, taus[i], lam=1e-5)
            y_hat_vector[j] = single_loss[0][0]
        temp = ((y_hat_vector.flatten()- test_y.flatten()) ** 2).mean()
        losses[i] = temp / 2.0
    
    return losses

def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO
    # first we divide all training examples into two sets, training set and 
    # validation set, using idx to define training and validation sets, since
    # idx is already randomized at the beginning in starter code
    total_num_training_examples = x.shape[0]
    total_num_validation = int(total_num_training_examples * val_frac)
    validation_set_idx = idx[0 : total_num_validation]
    
    validation_X = np.zeros((1, x.shape[1]))
    validation_target = np.array([[0]])
    training_X = np.zeros((1, x.shape[1]))
    training_target = np.array([[0]])
    
    for index in range(0, total_num_training_examples):
        if index in validation_set_idx:
            validation_X = np.vstack((validation_X, x[index, : ]))
            validation_target = np.vstack((validation_target, y[index]))
        else:
            training_X = np.vstack((training_X, x[index, : ]))
            training_target = np.vstack((training_target, y[index]))
            
    validation_X = validation_X[1 : , : ]
    validation_target = validation_target[1 : ]
    training_X = training_X[1 : , : ]
    training_target = training_target[1 : ]
    
    # validation set losses
    losses_for_validation = compute_losses(validation_X, validation_target, training_X, training_target, taus)
    losses_for_training = compute_losses(training_X, training_target, training_X, training_target, taus)
    
    return (losses_for_training, losses_for_validation)
    ## TODO

if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(taus, train_losses, color="red", label="Train Losses")
    plt.semilogx(taus, test_losses, color="blue", label="Validation (Test) Losses")
    #plt.plot(taus, train_losses, color="blue", label="Train Losses")
    #plt.plot(taus, test_losses, color="green", label="Validation (Test) Losses")
    plt.title('Training Losses/Validation (Test) Losses vs. Tau')
    plt.xlabel('Tau')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

