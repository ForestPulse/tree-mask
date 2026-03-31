#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import argparse
from sklearn import metrics 

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", 
                    help="path to the working directory", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_noFORCE")
parser.add_argument("--version", 
                    help="version of the model to be used",
                      default = '5')
args = parser.parse_args()

def norm(a):
    a_out = a/10000.
    return a_out

# =======================================
#              load data 
# =======================================
model_path = os.path.join(args.working_directory, '3_trained_model',
                            'version' +args.version, 'saved_model'+ args.version+ '.keras')
model = tf.keras.models.load_model(model_path)
x_val = np.load(os.path.join(args.working_directory, '2_train_test', 'test_x.npy'))
y_val= np.load(os.path.join(args.working_directory, '2_train_test', 'test_y.npy'))

# =======================================
#              prediction 
# =======================================
x_val = norm(x_val)
indices_vl = y_val - 1
# One-Hot-Encoding
y_val = np.eye(8)[indices_vl]
y_pred = model(x_val, training=False)
y_pred = y_pred.numpy()

# =======================================
#              evaluation 
# =======================================
validation_classes = np.argmax(y_val, axis=1)
prediction_classes = np.argmax(y_pred, axis=1)

confusion_matrix = metrics.confusion_matrix(validation_classes, prediction_classes) 
confusin_matrix_rel = np.round(confusion_matrix.astype('float') / np.array(confusion_matrix).sum() *100, 2)
sensitivity = metrics.recall_score(validation_classes, prediction_classes, average=None)
precision = metrics.precision_score(validation_classes, prediction_classes, average=None)
overall_accuracy = metrics.accuracy_score(validation_classes, prediction_classes)
f1_score = metrics.f1_score(validation_classes, prediction_classes, average=None)

classes = ['Artificial Land', 'Cropland', 'Woodland', 'Shrubland', 'Grassland', 'Bare Land', 'Water Areas', 'Wetlands']
n_classes = len(classes)

# =============================================
#              writing txt table 
# =============================================
if not os.path.exists( os.path.join(args.working_directory, '4_validation') ):
    os.makedirs(os.path.join(args.working_directory, '4_validation') ) 

with open(os.path.join(args.working_directory,"4_validation",
            "confusion_matrix_report_v{model_number}.txt".format(model_number=args.version)), "w") as f:
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
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
