import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras import losses
from keras import models
from keras import optimizers
from keras import callbacks
from keras import regularizers
from keras import activations
from keras import backend
from keras.callbacks import EarlyStopping
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
run_string = "cnn_base"
context_length = 18
train_test_simple = True

# parameters
size_encoding = 4  
sequence_length = 36
batch_size = 200
lr=0.001
hu_dense=20
epochs=5000
patience = 30


experiment_name = run_string + "_context:" + str(context_length)

# build paths for results
result_dir = "Results/" + experiment_name
summary_dir = os.path.join(result_dir, "summaries")
report_path = os.path.join(summary_dir, "test_results.txt")
best_model_path = os.path.join(result_dir, "best_model")
create_dir_if_not_exist([result_dir, summary_dir, best_model_path])

# load data
P_train = np.load("./../Dataset_Left_0_padding/P_Train.npy")
N_train = np.load("./../Dataset_Left_0_padding/N_Train.npy")
P_Test = np.load("./../Dataset_Left_0_padding/P_Test.npy")
N_Test = np.load("./../Dataset_Left_0_padding/N_Test.npy")

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
    # model.save(best_model_path + "/best_model")
    
    #define training set and target for the ensamble 
    predictions1 = model1.predict(X_train, batch_size=1, verbose=1)
    predictions2 = model2.predict(X_train, batch_size=1, verbose=1)
    predictions3 = model3.predict(X_train, batch_size=1, verbose=1)
    predictions4 = model4.predict(X_train, batch_size=1, verbose=1)
    predictions5 = model5.predict(X_train, batch_size=1, verbose=1)
    predictions6 = model6.predict(X_train, batch_size=1, verbose=1)
    predictions7 = model7.predict(X_train, batch_size=1, verbose=1)    
    mlp_train=np.array([predictions1[0][0], predictions2[0][0], predictions3[0][0], predictions4[0][0], predictions5[0][0], predictions6[0][0], predictions7[0][0]])
    for i in range(1, X_train.shape[0]):
       mlp_train=np.vstack([mlp_train, [predictions1[i][0], predictions2[i][0], predictions3[i][0], predictions4[i][0], predictions5[i][0], predictions6[i][0], predictions7[i][0]]])   
    mlp_train_t=Y_train
    
    #define valdation set and target for the ensamble
    val_predictions1 = model1.predict(X_val, batch_size=1, verbose=1)
    val_predictions2 = model2.predict(X_val, batch_size=1, verbose=1)
    val_predictions3 = model3.predict(X_val, batch_size=1, verbose=1)
    val_predictions4 = model4.predict(X_val, batch_size=1, verbose=1)
    val_predictions5 = model5.predict(X_val, batch_size=1, verbose=1)
    val_predictions6 = model6.predict(X_val, batch_size=1, verbose=1)
    val_predictions7 = model7.predict(X_val, batch_size=1, verbose=1)    
    mlp_val=np.array([val_predictions1[0][0], val_predictions2[0][0], val_predictions3[0][0], val_predictions4[0][0], val_predictions5[0][0], val_predictions6[0][0], val_predictions7[0][0]])
    for i in range(1, X_val.shape[0]):
        mlp_val=np.vstack([mlp_val, [val_predictions1[i][0], val_predictions2[i][0], val_predictions3[i][0], val_predictions4[i][0], val_predictions5[i][0], val_predictions6[i][0], val_predictions7[i][0]]])
    mlp_val_t=Y_val
    
    #define test set and target for the ensamble
    test_predictions1 = model1.predict(x_test, batch_size=1, verbose=1)
    test_predictions2 = model2.predict(x_test, batch_size=1, verbose=1)
    test_predictions3 = model3.predict(x_test, batch_size=1, verbose=1)
    test_predictions4 = model4.predict(x_test, batch_size=1, verbose=1)
    test_predictions5 = model5.predict(x_test, batch_size=1, verbose=1)
    test_predictions6 = model6.predict(x_test, batch_size=1, verbose=1)
    test_predictions7 = model7.predict(x_test, batch_size=1, verbose=1)
    mlp_test=np.array([test_predictions1[0][0], test_predictions2[0][0], test_predictions3[0][0], test_predictions4[0][0], test_predictions5[0][0], test_predictions6[0][0], test_predictions7[0][0]])
    for i in range(1, x_test.shape[0]):
        mlp_test=np.vstack([mlp_test, [test_predictions1[i][0], test_predictions2[i][0], test_predictions3[i][0], test_predictions4[i][0], test_predictions5[i][0], test_predictions6[i][0], test_predictions7[i][0]]])
    mlp_test_t=y_test
    
    
    #build the model
    print("Building mlp model")
    # build optimizer instance
    optimizer = optimizers.Adam(lr)
    stopper = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto', restore_best_weights = True)
    #build the layers
    layer_input = layers.Input(shape=(mlp_train.shape[1],))
    classification_layer = layers.Dense(units=hu_dense, activation='relu')(layer_input) 
    layer_output = layers.Dense(units=2, activation='softmax')(classification_layer)
    #compile the model
    model = models.Model(inputs = layer_input, outputs = layer_output)
    model.compile(loss=losses.categorical_crossentropy, optimizer = optimizer, metrics=['accuracy'])
    
    #print the model summary
    model.summary()
    
    #train the model
    print("Training the model")
    model.fit(mlp_train, mlp_train_t, epochs=epochs, callbacks=[stopper], batch_size=1, validation_data=(mlp_val, mlp_val_t), verbose=1)
    
    # binarize predictions and encode labels
    mlp_predictions=model.predict(mlp_test,batch_size=1, verbose=1)
    mlp_test_t = np.argmax(mlp_test_t, axis=1)
    mlp_predictions = np.argmax(mlp_predictions, axis=1)

    report = "Test on:" + str(np.shape(mlp_test)) + " examples \n"
    report = report + classification_report(mlp_test_t, mlp_predictions, target_names=["lente", "veloci"])
    report = report + "\n accuracy: " + str(accuracy_score(mlp_test_t, mlp_predictions))

    file_object = open(report_path, 'w')
    file_object.write(report)
    file_object.close()

    print("TEST RESULTS: \n")
    print(report)
    
    
    
    
    
    
