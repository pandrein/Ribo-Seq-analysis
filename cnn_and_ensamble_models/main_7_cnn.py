# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:21:38 2021

@author: Caterina
"""


import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras import losses
from keras import optimizers
from keras import models
from keras import callbacks
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from utils import create_dir_if_not_exist
from utils import extract_dataset
from utils import clear_folder
from utils import summary_to_file
from NetworkModels import NetworkModel
import talos
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# command line arguments
run_string = "ensamble_7cnn_base"
context_length = 18
train_the_model_grid_search = False
test_the_model_grid_search = False
train_test_simple = True

# parameters
size_encoding = 4  # size of one-hot amino acid encoding 4 nucleotides, 12 codon
sequence_length = 36  # length of the sequences in the dataset 36 nucleotides, 13 codon
batch_size = 200
lr=0.001
hu_dense=20
epochs=5000
patience = 30

experiment_name = run_string + "_context_" + str(context_length)

# build paths for results
result_dir = "Results/" + experiment_name
summary_dir = os.path.join(result_dir, "summaries")
report_path = os.path.join(summary_dir, "test_results.txt")
best_model_path = os.path.join(result_dir, "best_model")
create_dir_if_not_exist([result_dir, summary_dir, best_model_path])

# load data
P_train = np.load("./datasets/padding0/P_Train.npy")
N_train = np.load("./datasets/padding0/N_Train.npy")
P_Test = np.load("./datasets/padding0/P_Test.npy")
N_Test = np.load("./datasets/padding0/N_Test.npy")

x_train, y_train, x_test, y_test = extract_dataset(N_train, P_train, N_Test, P_Test, sequence_length, size_encoding, context_length)

print("Dataset loaded - Training shape: " + str(np.shape(x_train)) + "Test shape: " + str(np.shape(x_test)))

if train_test_simple:
    neural_network = NetworkModel(context_length, sequence_length, size_encoding, best_model_path, batch_size=batch_size)
    params = {'layer1_units': 16, 'layer2_units': 16, 'hu_dense': 16, 'epochs': 5000, 'dropout': 0.5}
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=44)
    history, model1 = neural_network.train_simple_cnn_model(X_train, Y_train, X_val, Y_val, params)
    
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=33)
    history, model2 = neural_network.train_simple_cnn_model(X_train, Y_train, X_val, Y_val, params)
        
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=22)
    history, model3 = neural_network.train_simple_cnn_model(X_train, Y_train, X_val, Y_val, params)
    
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=11) 
    history, model4 = neural_network.train_simple_cnn_model(X_train, Y_train, X_val, Y_val, params)
    
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=55)
    history, model5 = neural_network.train_simple_cnn_model(X_train, Y_train, X_val, Y_val, params)
    
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=66) 
    history, model6 = neural_network.train_simple_cnn_model(X_train, Y_train, X_val, Y_val, params)
        
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=77) 
    history, model7 = neural_network.train_simple_cnn_model(X_train, Y_train, X_val, Y_val, params)

    predictions1 = model1.predict(x_test, batch_size=1, verbose=1)
    predictions2 = model2.predict(x_test, batch_size=1, verbose=1)
    predictions3 = model3.predict(x_test, batch_size=1, verbose=1)
    predictions4 = model4.predict(x_test, batch_size=1, verbose=1)
    predictions5 = model5.predict(x_test, batch_size=1, verbose=1)
    predictions6 = model6.predict(x_test, batch_size=1, verbose=1)
    predictions7 = model7.predict(x_test, batch_size=1, verbose=1)

            
    predictions = np.sum([predictions1,predictions2,predictions3,predictions4,predictions5,predictions6,predictions7], axis=0) #, predictions_xgb],

    ##binarize predictions and encode labels
    y_test = np.argmax(y_test, axis=1)
    predictions = np.argmax(predictions, axis=1)

    report = "Test on:" + str(np.shape(x_test)) + " examples \n"
    report = report + classification_report(y_test, predictions, target_names=["lente", "veloci"])  # FIX ME potrebbero essere invertite...
    report = report + "\n accuracy: " + str(accuracy_score(y_test, predictions))


    file_object = open(report_path, 'w')
    file_object.write(report)
    file_object.close()

    print("TEST RESULTS: \n")
    print(report)

