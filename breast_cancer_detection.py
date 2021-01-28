#!/usr/bin/env python
# coding: utf-8

# #  Breast Cancer Detection using Deep Neural Network 
# 
# #### <font color='darkblue'> Mekelle University Contract Research and Publications Team ( Oct,2017 - Sep, 2018)<font>
# 
#  1. Mr. Hadush Hailu (BSc.) Principal Investigator -- hadushhailugeb@gmail.com
#  2. Ms. Mhret Berhe (MSc. Student) Co- Investigator
#  3. Mr. Simon Hadush (Msc.student) Co- Investigator
#  4. Mr. Solomun Kahsay(MSc) Co- Investigator
#  5. Mr. Tedros Hailu (Undergraduate student) Member
# 
# #### <font color='darkblue'> Maintained by: Hadush Hailu ( Sep, 2018 - )<font>
#     
# #### Short summary
# Breast cancer has become the leading cause of cancer deaths among women. To decrease the related
# Mortality, disease must be treated early, but it is hard to detect and diagnose tumors at an early stage.
# What is currently being employed is manual diagnosis of each image pattern by a professional
# radiologist. This manual diagnosis and detection of breast cancer have proven to be time taking and
# inefficient in many cases. Mostly those techniques are imperfect and often result in false positive results
# that can lead to unnecessary biopsies and Surgeries. This means that every year thousands of women go
# through painful, expensive, scar-inducing surgeries that weren’t even necessary. Hence there is a need
# for efficient methods that diagnoses the cancerous cell without involvement of humans with high
# accuracy.
# This research proposes an automated technique using Deep  Neural Network as decision making tools
# in the field of breast cancer. Image Processing plays significant role in cancer detection when input data
# is in the form of images. Feature extraction (statistical parameter) of image is important in mammogram
# classification. Features are extracted by using image processing. Different feature extraction methods
# used for classification of normal and abnormal patterns in mammogram. This method will give
# maximum accuracy at a high speed. The statistical parameter include entropy, mean, correlation,
# standard deviation .This parameters will act as a inputs to Deep Neural Network( here after Deep NN)
# which will diagnose and give the result whether image is cancerous and non-cancerous.
# 

# ## 1 Packages
# 
# ### List of packages we have used in this project
# 
# - numpy is the main package for scientific computing with Python.
# - matplotlib is a library to plot graphs in Python.
# - pandas for data processing
# - sklearn for dataset test train split

# In[778]:


#### Important packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Dataset  information and Preprocessing
# ##### ( for more information: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
# Data Set Characteristics:  
#     - Multivariate: Multivariate
#     - Number of Instances: 569
#     - Number of Attributes: 32
#     - Date Donated: 1995-11-01
#     - Associated Tasks: Classification
#     - Missing Values? No
#     
# 
# Attribute Information:
# 
# 1. ID number 
# 2. Diagnosis (M = malignant, B = benign) 
# 3. 3 -32 
# 
# Ten real-valued features are computed for each cell nucleus: 
# 
#     a) radius (mean of distances from center to points on the perimeter) 
#     b) texture (standard deviation of gray-scale values) 
#     c) perimeter 
#     d) area 
#     e) smoothness (local variation in radius lengths) 
#     f) compactness (perimeter^2 / area - 1.0) 
#     g) concavity (severity of concave portions of the contour) 
#     h) concave points (number of concave portions of the contour) 
#     i) symmetry 
#     j) fractal dimension ("coastline approximation" - 1)
# 
# 

# In[779]:


#### Read csv data using Pandas

colnames = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean',
       'concavity_mean', 'concave points_mean', 'symmetry_mean',
       'fractal_dimension_mean', 'radius_se', 'texture_se',
       'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
       'concavity_se', 'concave points_se', 'symmetry_se','fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']


# In[780]:



def data_pre_process(data,X_ini_col, X_fin_col,Y_col):
    
    X = data[:,X_ini_col: X_fin_col].astype(float) # remove the 'id' column and turn the value into float
    Y = np.where(data[:,Y_col:Y_col+1] == 'M', 1,0)
    
    # Split the data given as (X,Y) to train data and test data
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    # Transpose the data so that it would be column wise and normalize it
    x_train = x_train.T
    x_train = (x_train - np.mean(x_train,axis=0))/np.std(x_train,axis=0) 
    
    x_test = x_test.T
    x_test = (x_test - np.mean(x_test,axis=0))/np.std(x_test,axis=0)
    
    y_train = y_train.T
    y_test = y_test.T
    
    return x_train,x_test,y_train,y_test



# ## 3 Intialize parameters
# 
#     num_input -- size of the input layer
#     num_hidden -- size of the hidden layer
#     num_output -- size of the output layer

# In[781]:


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims - python list containing the dimensions of each layer in the network
    
    Returns:
    parameters - python dictionary containing the parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        
        # Multiplying weight by 0.01 prevents from exploding gradients.
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


# ## 4 Forward propagation module

# In[782]:


def sigmoid(Z):
    a = 1/(1 + np.exp(-Z))
    return a

def relu(Z):
    a = np.maximum(0,Z)
    return a


# In[783]:


def forward_pre_activated(A, W, b):
    
    """
    Arguments: A, W, b

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache_pre_activated -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W,A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache_pre_activated = (A, W, b)
    
    return Z, cache_pre_activated


# In[784]:


def forward_activated(A_prev, W, b, activation):
    """
    Arguments: A_prev, W, b, activation 
    
    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache_activated -- a python tuple containing "cache_pre_activated" and "Z";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        Z, cache_pre_activated = forward_pre_activated(A_prev,W,b)
        A = sigmoid(Z)
    
    elif activation == "relu":
        Z, cache_pre_activated = forward_pre_activated(A_prev,W,b)
        A = relu(Z)
        
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache_activated = (cache_pre_activated, Z)

    return A, cache_activated


# In[785]:


def deep_forward_module(X, parameters):
    """
    Arguments: X, parameters
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of cache_activated (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
   
    for l in range(1, L):
        # layers 0 - l-1
        A_prev = A 
        A, cache = forward_activated(A_prev,parameters['W' + str(l)],parameters['b' + str(l)],"relu")
        caches.append(cache)
        
    
    # The last l layer
    AL, cache = forward_activated(A,parameters['W' + str(L)],parameters['b' + str(L)],"sigmoid")
    caches.append(cache)
    
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


# ## 5  Computing Cross-Entropy

# In[786]:


def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = - np.sum(Y*np.log(AL) + (1 -Y)*np.log(1-AL))/m
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


# ## 6 Backward propagation module

# In[787]:


def backward_pre_activated(dZ, cache):
    """
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis = 1,keepdims = True)/m
    dA_prev = np.dot(W.T,dZ)
   
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


# In[788]:


def backward_activated(dA, cache, activation):
    """
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    cache_pre_activated, Z = cache
    
    if activation == "relu":
        drelu = np.where(Z > 0, 1, 0)
        dZ = np.multiply(dA,drelu)
        dA_prev, dW, db = backward_pre_activated(dZ,cache_pre_activated)
       
        
        
    elif activation == "sigmoid":
        dsigmoid = np.multiply(sigmoid(Z), (1- sigmoid(Z)))
        dZ = np.multiply(dA,dsigmoid)
        dA_prev, dW, db = backward_pre_activated(dZ,cache_pre_activated)
    
    return dA_prev, dW, db


# In[789]:


def deep_backward_module(AL, Y, caches):
    """    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    gradiants = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y,AL)) + (np.divide(1 - Y, 1 - AL))
    #dAL = np.divide(AL - Y, (1 - AL) *  AL)
   
    
    # Lth layer: sigmoid activation function
    current_cache = backward_activated(dAL, caches[L-1], "sigmoid")
    gradiants["dA" + str(L-1)], gradiants["dW" + str(L)], gradiants["db" + str(L)] = current_cache[0],current_cache[1],current_cache[2]
  
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: relu activation function
        current_cache = backward_activated(gradiants["dA" + str(l+1)], caches[l], "relu")
        dA_prev_temp, dW_temp, db_temp = current_cache[0],current_cache[1],current_cache[2]
        gradiants["dA" + str(l)] = dA_prev_temp
        gradiants["dW" + str(l + 1)] = dW_temp
        gradiants["db" + str(l + 1)] = db_temp
        
    return gradiants


# ## 7. Update Parameters

# In[790]:


def update_parameters(parameters, gradiants, learning_rate):
    """ 
    Arguments:
    parameters -- python dictionary containing your parameters 
    gradiants -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * gradiants["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * gradiants["db" + str(l+1)]

    return parameters


# ## 8 Complete Model

# In[791]:


def deep_nn_model(x_train,y_train,layer_dims, num_iterations = 10000, print_cost=False):
    """
    Arguments: x_train,y_train,layer_dims,num_iterations,
   
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(1)
   
    # Initialize parameters
    parameters = initialize_parameters(layer_dims)
    

    # Loop (gradient descent)
    for i in range(0, num_iterations):
         
        # run deep forward module
        AL, caches =  deep_forward_module(x_train,parameters)

        # cost
        cost = compute_cost(AL,y_train)

        # run deep backward module
        gradiants = deep_backward_module(AL, y_train, caches)

        # update parameters
        parameters = update_parameters(parameters, gradiants, 0.08)
    
        
        # Print the cost every 1000 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


# ## 9 Predictions

# In[792]:


def predict(parameters, X):
    """  
    Arguments:
    parameters, X 
    
    Returns
    predictions 
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    AL, cache = deep_forward_module(X,parameters)
    predictions = np.where(AL>0.5,1,0)
    
    return predictions


# ### Performance

# In[793]:


### read data
print("-------------------------------------")
print("Breast Cancer Detection using Deep NN")
print("-------------------------------------")
df = pd.read_csv('data.csv')
x_train,x_test,y_train,y_test = data_pre_process(df.values,2,32,1)

print("Dataset:  Breast Cancer Wisconsin (Diagnostic) Data Set")
print("data Split: x_train:",x_train.shape,"x_test:",x_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape)
print(" ")
### decide model architecture
layer_dims = [30,9,1]

### Run model
parameters = deep_nn_model(x_train,y_train,layer_dims)

### Model Accuracy
predictions = predict(parameters, x_test)
print ('Accuracy: %d' % float((np.dot(y_test,predictions.T) + np.dot(1-y_test,1-predictions.T))/float(y_test.size)*100) + '%')
