import os
import pandas as pd
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from keras import losses
from keras import optimizers
from keras import models
from keras import layers
from keras import callbacks
from keras import regularizers
from keras import activations
from keras import backend
from keras import models
from talos.model.early_stopper import early_stopper
# from keras.callbacks import EarlyStopping, ModelCheckpoint
import talos
import tensorflow as tf
# earlyStopping = EarlyStopping(patience=0, verbose=0, mode='min', restore_best_weights=True)

# command line arguments
run_string = "nucleotides_mlp"  # sys.argv[1]  # string to discriminate results of different runs
context_length = 18  # int(sys.argv[2]) # the max context_length is 6 for codons 18 for nucleotides
train_the_model = True
test_the_model = True

# parameters
size_encoding = 4  # size of one-hot amino acid encoding 4 nucleotides, 12 codon
classes = 2  # number of classes (positive / negative)
sequence_length = 36  # length of the sequences in the dataset 36 nucleotides, 13 codon
encoding = "onehot"

# hyperparameters
batch_size = 5000  # size of training batch 
class_weights = [1, 1]

p = {'slr': [0.001],
     'hu_dense': [10,20,40,80,160, 200, 300, 400, 500, 600, 700, 800, 900],
     'epochs': [5000],
     }

experiment_name = "_encoding" + encoding + "_type-" + run_string + "_context" + str(context_length)

# build paths for results
fold_results_dir = "0_padding/Results/" + experiment_name + "/fold_by_fold/" + experiment_name + "/"
summary_path = "0_padding/Results/" + experiment_name + "/summaries/" + experiment_name + ".txt"
best_model_path = "0_padding/Results/" + experiment_name + "/best_model/"

# build directory system for run results
if not os.path.exists("0_padding/Results/" + experiment_name):
    os.makedirs("0_padding/Results/" + experiment_name)
if not os.path.exists("0_padding/Results/" + experiment_name + "/summaries"):
    os.makedirs("0_padding/Results/" + experiment_name + "/summaries")
if not os.path.exists("0_padding/Results/" + experiment_name + "/fold_by_fold"):
    os.makedirs("0_padding/Results/" + experiment_name + "/fold_by_fold")
if not os.path.exists("0_padding/Results/" + experiment_name + "/fold_by_fold/" + experiment_name):
    os.makedirs("0_padding/Results/" + experiment_name + "/fold_by_fold/" + experiment_name)
if not os.path.exists(best_model_path):
    os.makedirs(best_model_path)


def extract_dataset(negative, positive):
    # slice previous context and sequence to desired context length
    index_start = int(sequence_length / 2 - context_length)
    index_stop = index_start + 2 * context_length

    positive = positive[:, index_start:index_stop + sequence_length % 2, 0:size_encoding]  # 0:4 nucleotides
    negative = negative[:, index_start:index_stop + sequence_length % 2, 0:size_encoding]

    y_pos_p0 = np.zeros((positive.shape[0], 2 * context_length + sequence_length % 2, 1), dtype=np.float64)

    y_pos_p1 = np.ones((positive.shape[0], 2 * context_length + sequence_length % 2, 1), dtype=np.float64)
    positive = np.concatenate((positive, y_pos_p0, y_pos_p1), axis=2)
    y_neg_p0 = np.ones((negative.shape[0], 2 * context_length + sequence_length % 2, 1), dtype=np.float64)
    y_neg_p1 = np.zeros((negative.shape[0], 2 * context_length + sequence_length % 2, 1), dtype=np.float64)
    negative = np.concatenate((negative, y_neg_p0, y_neg_p1), axis=2)

    return np.concatenate((positive, negative), axis=0)


# load data
print("Loading input data")
positive = None  # positive input tensor
negative = None  # negative input tensor

P_train = np.load("./Dataset_Left_0_padding/P_Train.npy")
N_train = np.load("./Dataset_Left_0_padding/N_Train.npy")
P_Test = np.load("./Dataset_Left_0_padding/P_Test.npy")
N_Test = np.load("./Dataset_Left_0_padding/N_Test.npy")

# P_train = np.load("./dataset_splitted/seqs2datasets_N2C/Datasets/RibProf_Seqs/NL36m6M18_CL13/CodonRelaxed/Left/P_Train.npy")
# N_train = np.load("./dataset_splitted/seqs2datasets_N2C/Datasets/RibProf_Seqs/NL36m6M18_CL13/CodonRelaxed/Left/N_Train.npy")
# P_Test = np.load("./dataset_splitted/seqs2datasets_N2C/Datasets/RibProf_Seqs/NL36m6M18_CL13/CodonRelaxed/Left/P_Test.npy")
# N_Test = np.load("./dataset_splitted/seqs2datasets_N2C/Datasets/RibProf_Seqs/NL36m6M18_CL13/CodonRelaxed/Left/N_Test.npy")

# print(np.shape(P_train))

Xy_tr = extract_dataset(N_train, P_train)
Xy_test = extract_dataset(N_Test, P_Test)

np.random.shuffle(Xy_tr)
np.random.shuffle(Xy_test)

# split test target and data tensors again
Xc = Xy_test[:, :, :size_encoding]
yc = Xy_test[:, 0, size_encoding:]

Fc = np.sum(Xc, axis=1)  # absolute frequency in Test
i = 0
FTestlist = list()  # relative frequencies list
for seq in Xc:
    len = np.sum(seq)
    FTestlist.append(1 / len * Fc[i])
    i+=1
FTest = np.array(FTestlist)

# declare global statistics
GLOBAL_TP = 0
GLOBAL_TN = 0
GLOBAL_FP = 0
GLOBAL_FN = 0

# split target and data tensors
Xa = Xy_tr[:, :, :size_encoding]
ya = Xy_tr[:, 0, size_encoding:]

Fa = np.sum(Xa, axis=1)  # absolute frequency Training

j = 0
FTrainlist = list()  # relative frequencies list
for seq in Xa:
    len = np.sum(seq)
    FTrainlist.append(1 / len * Fa[j])
    j += 1
FTrain = np.array(FTrainlist)


def mlp(x_train, y_train, x_val, y_val, params):
    # build the model
    print("Building mlp model")
    # build optimizer instance
    optimizer = tf.optimizers.Adam(lr=params['slr'])
    # build the model
    layer_input = layers.Input(shape=(size_encoding,))

    classification_layer = layers.Dense(units=int(params['hu_dense']), activation='relu')(layer_input)  # PAOLO mlp di classificazione
    layer_output = layers.Dense(units=2, activation='softmax')(classification_layer)

    model = models.Model(inputs=layer_input, outputs=layer_output)

    print(model.summary())
    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    # train the model
    print("Training the model")
    from keras.callbacks import ModelCheckpoint

    # out = model.fit(x_train, y_train, epochs=params['epochs'], callbacks=[earlyStopping], batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)
    out = model.fit(x_train, y_train, epochs=params['epochs'], callbacks=[early_stopper(epochs=params['epochs'], patience=0), ModelCheckpoint(
        filepath=best_model_path,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)], batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)
    return out, model




if train_the_model:
    print(type(FTrain))
    print(np.shape(FTrain))
    scan_object = talos.Scan(FTrain, ya, params=p, model=mlp, experiment_name="param_selection" + experiment_name, val_split=0.30, seed=1)

    print("param_selection" + experiment_name)

    model = scan_object.best_model(metric='val_accuracy', asc=False)

    model.save(best_model_path + "best_model")


if test_the_model:
    model = models.load_model(best_model_path + "best_model")

    # Make a report on the scan
    exts = tuple(["csv"])
    report_csv_path = [os.path.abspath(os.path.join(os.getcwd() + "/" + "param_selection" + experiment_name, f)) for f in os.listdir(os.getcwd() + "/" + "param_selection" + experiment_name) if f.endswith(exts)][-1]

    report = talos.Reporting(report_csv_path)
    best_model_params = report.data.sort_values('val_accuracy').tail(1)
    slr = best_model_params['slr'].to_numpy()
    hu_dense = best_model_params['hu_dense'].to_numpy()
    epochs = best_model_params['epochs'].to_numpy()
    val_accuracy = best_model_params['val_accuracy'].to_numpy()
    epochs = best_model_params['round_epochs'].to_numpy()

    predictions = model.predict(FTest, batch_size=batch_size, verbose=1)
    #
    # evaluation of results
    print("Evaluating model performances")
    TP, TN, FP, FN = 0, 0, 0, 0
    # compare predicted target tensor to real target tensor, counting true/false positive/negative examples
    for i in range(yc.shape[0]):
        if yc[i][1] >= 0.5:
            if predictions[i][1] >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            if predictions[i][1] >= 0.5:
                FP += 1
            else:
                TN += 1
    # calculate evaluation metrics
    precision = None
    recall = None
    f1score = None
    if TP + FP != 0:
        precision = float(TP) / float(TP + FP)
    if TP + FN != 0:
        recall = float(TP) / float(TP + FN)
    if precision is not None and recall is not None:
        f1score = 2 * precision * recall / (precision + recall)
    accuracy = float(TP + TN) / float(TP + TN + FP + FN)

    #
    # add local statistics to global statistics
    GLOBAL_TP = GLOBAL_TP + TP
    GLOBAL_TN = GLOBAL_TN + TN
    GLOBAL_FP = GLOBAL_FP + FP
    GLOBAL_FN = GLOBAL_FN + FN

    # compute average values of the statistics over the ten folds
    GLOBAL_precision = None
    #
    # evaluation of results
    print("Evaluating model performances")
    TP, TN, FP, FN = 0, 0, 0, 0
    # compare predicted target tensor to real target tensor, counting true/false positive/negative examples
    for i in range(yc.shape[0]):
        if yc[i][1] >= 0.5:
            if predictions[i][1] >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            if predictions[i][1] >= 0.5:
                FP += 1
            else:
                TN += 1
    # calculate evaluation metrics
    precision = None
    recall = None
    f1score = None
    if TP + FP != 0:
        precision = float(TP) / float(TP + FP)
    if TP + FN != 0:
        recall = float(TP) / float(TP + FN)
    if precision is not None and recall is not None:
        f1score = 2 * precision * recall / (precision + recall)
    accuracy = float(TP + TN) / float(TP + TN + FP + FN)

    #
    # add local statistics to global statistics
    GLOBAL_TP = GLOBAL_TP + TP
    GLOBAL_TN = GLOBAL_TN + TN
    GLOBAL_FP = GLOBAL_FP + FP
    GLOBAL_FN = GLOBAL_FN + FN

    # compute average values of the statistics over the ten folds
    GLOBAL_precision = None
    GLOBAL_recall = None
    GLOBAL_f1score = None
    if GLOBAL_TP + GLOBAL_FP != 0:
        GLOBAL_precision = float(GLOBAL_TP) / float(GLOBAL_TP + GLOBAL_FP)
    if GLOBAL_TP + GLOBAL_FN != 0:
        GLOBAL_recall = float(GLOBAL_TP) / float(GLOBAL_TP + GLOBAL_FN)
    if GLOBAL_precision is not None and GLOBAL_recall is not None:
        GLOBAL_f1score = 2 * GLOBAL_precision * GLOBAL_recall / (GLOBAL_precision + GLOBAL_recall)
    GLOBAL_accuracy = float(GLOBAL_TP + GLOBAL_TN) / float(GLOBAL_TP + GLOBAL_TN + GLOBAL_FP + GLOBAL_FN)

    # print aggregated results
    print("Printing results to " + summary_path)
    out_file = open(summary_path, 'w')
    out_file.write("HYPERPARAMETERS\n")
    out_file.write("Dense hidden units : " + str(hu_dense) + "\n")
    out_file.write("Starting learning rate : " + str(slr) + "\n")
    out_file.write("Batch size : " + str(batch_size) + "\n")
    out_file.write("Class weights : " + str(class_weights) + "\n")
    out_file.write("Epochs : " + str(epochs) + "\n")
    out_file.write("\n")
    out_file.write("RESULTS\n")
    out_file.write("Validation accuracy : " + str(val_accuracy) + "\n")
    out_file.write("TP : " + str(GLOBAL_TP) + "\n")
    out_file.write("TN : " + str(GLOBAL_TN) + "\n")
    out_file.write("FP : " + str(GLOBAL_FP) + "\n")
    out_file.write("FN : " + str(GLOBAL_FN) + "\n")
    out_file.write("Precision : " + str(GLOBAL_precision) + "\n")
    out_file.write("Recall : " + str(GLOBAL_recall) + "\n")
    out_file.write("F1-score : " + str(GLOBAL_f1score) + "\n")
    out_file.write("Accuracy : " + str(GLOBAL_accuracy) + "\n")
    # out_file.write("Global average attention : "+str(GLOBAL_average_attention))
    out_file.close()

    # terminate execution
    print("Average Precision : " + str(GLOBAL_precision))
    print("Average Recall : " + str(GLOBAL_recall))
    # print("Average F1-score : "+str(GLOBAL_f1score))
    print("Average Accuracy : " + str(GLOBAL_accuracy))


