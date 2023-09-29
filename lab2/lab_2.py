# 4C16 Lab 2 -- logistic regression

import numpy as np
import math
# The core of logistic regression is the sigmoid function, which maps
# the real line to the range 0--1 (loosely, a probability).
#
# The usual sigmoid is the logistic function
# sigmoid(t)=1/(1 + exp(-t)).
#
# However this is prone to numerical overflow when 't' is large.  So
# we are using tanh as our sigmoid, to avoid this problem.
#
# Tanh produces values in the range -1..1, so we add 1 and divide by 2
# to map this to the desired range 0..1.
#
# We divide the input value by two so that the slope of the linear
# part of the curve matches the logistic function.
def sigmoid(t):
    return (np.tanh(t/2) + 1)/2

# # Computing the risk score
def logit(w, X):
    return np.dot(X, w)

# # Prediction: apply the sigmoid to the risk scores.
def predict(w, X):
    return sigmoid(logit(w,X))


#
# #### EXERCISE 1 ####
#
# Fill out this function to compute the average cross entropy, E(w)/n,
#   with E(w) is as defined on page 20 handout 2  Note: this is a typo in the handout!
#
#   'w' are the weights
#   'X' are the observations
#   'y' are the true class values
def cross_entropy(w, X, y):
    # Use the 'predict' function to compute the predicted probability of label 1
    p = predict(w, X)

    # Now compute the cross entropy.
    #
    # Because this involves taking logs, you should add 'eps' where necessary to
    # avoid taking the log of 0.
    eps = 0.000001

    # Computation of the cross-entropy can be done in one line using numpy
    # functions log and mean.
    
    # Or it can be done in a more straightforward way: initialize an accumulator
    # variable to 0, do a 'for' loop over the elements of 'y', and update the
    # accumulator as appropriate (using 'math.log'). (THIS WILL BE SLOW)

    loss = -y * np.log(p + eps) - (1 - y) * np.log(1 - p + eps)
    return np.mean(loss)


#
# #### EXERCISE 2 ####
#
# Fill out this function to compute the gradient
#
# w: weight parameters
# X: design matrix containing the features for all observations
# y: the vector of the outcomes (a vector of booleans)
#
# Note that we return here the gradient *averaged* over all the observations, which
# differs slightly from the definition in the notes.
def gradient(w, X, y):
    n = y.shape[0]                 # number of observations
    p = predict(w, X)                    # <replace '[0]' with a call to the predict function>
    grad = np.dot(X.T, p - y)           # use 'np.dot' to compute the vector
    return grad / n                     # return the average gradient
#
# #### EXERCISE 3 ####
#
# Quiz-style: just return the number corresponding to your answer.
#
# What learning rate is best for the data set supplied in the notebook?
# eg. return 45 or return 0.1 (doesn't have to be super precise)
def question_3():
    return 0 # <- change this value

#
# #### EXERCISE 4 ####
#
# Write a function predict_class which uses weights 'w', observations
# 'X', and a threshold 't' to classify the data.
def predict_class(w, X, t):
    # replace with a vector of comparisons of a call to predict with 't':
    return np.zeros((X.shape[0],1))
    

#
# #### EXERCISE 5 ####
#
# Quiz-style: just return the number corresponding to your answer.
#
# What is the accuracy of your classifier for a threshold of 0.75

def question_5():
    return 0 # <- update this with actual accuracy

#
# #### EXERCISE 6 ####
#
# Quiz-style: return the bias in weight vector associated with logit of class=0
#
# hint: biases can be obtained from log_reg.intercept_ 
#       and the other coefficients from log_reg.coef_.

def question_6():
    return 0 # <- change this with estimated bias for logit of class=0

# #### EXERCISE 7 ####
# 
# in this exercise, you must find the best pair of features in the iris dataset.
# To find this pair, you will consider every possible pair and reduce the 
# input features to only that pair (and ignore the other two features). 
# For each of these pairs, you will thus need to 
# 1. modify both training and test sets to only include the two considered features
# 2. train a multinomial logistic regression model based on this reduced feature set
# 3. make prediction on the reduced test set 
# 4. report accuracy for that pair
#
# then change question_7 to return correct information about that best pair.
# For instance, if best pair is (0, 1), which gives you 77.77% accuracy, 
# your answer will be:
#    return {
#      'Features': (0,1), # replace with the best pair
#      'Acc': 0.77777     # replace with the accuracy for that pair
#      }   
#
# !! The results are sensitive to the random split and optimisation. 
# Thus make sure that the random_state is kept to random_state = 11 everywhere,
# and that the split kept to 0.3. (ie. keep original cells)

def question_7():
    return {
      'Features': (0,1), # replace with the best pair
      'Acc': 0.77777     # replace with the accuracy for that pair
      }    







