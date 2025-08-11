#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import argparse
from sklearn import metrics 

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the working directory", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_LCC")
args = parser.parse_args()

def norm(a):
    a_out = a/10000.
    return a_out

# load model
model_number = 3
epoch_number = 900
model_path = os.path.join(args.working_directory, '6_trained_model','version' +str(model_number), 'saved_model'+ str(model_number)+ '.keras')
#model_path = os.path.join(args.working_directory, '6_trained_model','version' +str(model_number), 'saved_model'+ str(model_number)+ '_epoch'+str(epoch_number)+'.keras')
model = tf.keras.models.load_model(model_path)

print('Evaluate model on validation data')
x_val = np.load(os.path.join(args.working_directory, '5_folded_data', 'test_augmented_folded_x.npy'))
x_val = norm(x_val)
y_val= np.load(os.path.join(args.working_directory, '3_train_test_data', 'test_y.npy'))
indices_vl = y_val - 1
# One-Hot-Encoding
y_val = np.eye(8)[indices_vl]

y_pred = model(x_val, training=False)
y_pred = y_pred.numpy()

validation_classes = np.argmax(y_val, axis=1)
prediction_classes = np.argmax(y_pred, axis=1)

confusion_matrix = metrics.confusion_matrix(validation_classes, prediction_classes) 
print(confusion_matrix)
print('Producers accuracy / Sesitivity:')
sensitivity = metrics.recall_score(validation_classes, prediction_classes, average=None)
print(sensitivity)
print('Users accuracy / Precision:')
precision = metrics.precision_score(validation_classes, prediction_classes, average=None)
print(precision)
print('Overall accuracy:')
overall_accuracy = metrics.accuracy_score(validation_classes, prediction_classes)
print(overall_accuracy)
print('F1 score:')
f1_score = metrics.f1_score(validation_classes, prediction_classes, average=None)
print(f1_score)

classes = ['Artificial Land', 'Cropland', 'Woodland', 'Shrubland', 'Grassland', 'Bare Land', 'Water Areas', 'Wetlands']
n_classes = len(classes)
# Formatieren und Speichern

if not os.path.exists( os.path.join(args.working_directory, '7_validation') ):
    os.makedirs(os.path.join(args.working_directory, '7_validation') ) 

with open(os.path.join(args.working_directory,"7_validation","confusion_matrix_report.txt"), "w") as f:
#with open(os.path.join(args.working_directory,"7_validation","confusion_matrix_report"+ '_epoch'+str(epoch_number)+".txt"), "w") as f:
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
    # Header
    header = "{:<16}".format(" ") + "".join(["{:>12}".format(c[:10]) for c in classes])
    header += "{:>17}{:>17}{:>17}".format("Sensitivity/PA", "Precision/UA", "F1-Score")
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")

    # Jede Zeile mit Werten
    for i in range(n_classes):
        row = "{:<16}".format(classes[i])
        for j in range(n_classes):
            row += "{:>12}".format(confusion_matrix[i, j])
        row += "{:>17.2f}{:>17.2f}{:>17.2f}".format(sensitivity[i], precision[i], f1_score[i])
        f.write(row + "\n")
