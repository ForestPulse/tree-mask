#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import os
import argparse
#from tensorflow.keras.callbacks import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", 
                    help="path to the working directory", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/tree_mask_noFORCE")
args = parser.parse_args()

def train(model_number):
    def norm(a):
        a_out = a/10000.
        return a_out

    def get_MLP_model(input_shape, lc_num):
        x_in = tf.keras.Input(shape=(input_shape,))
        x = tf.keras.layers.Flatten()(x_in)

        x = tf.keras.layers.Dense(96)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x_out = tf.keras.layers.Dense(lc_num, activation="softmax")(x)

        model = tf.keras.Model(inputs = x_in, outputs = x_out)
        return model
    
    def get_loss(x, y, model, training=True):
        y_pred = model(x, training=training)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, y_pred))
        return loss
    
    @tf.function
    def train_step(x, y, model, opt):
        with tf.GradientTape() as tape:
            loss = get_loss(x, y, model, training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    x_train= np.load(os.path.join(args.working_directory, '2_train_test', 'train_x.npy'))
    y_train= np.load(os.path.join(args.working_directory, '2_train_test', 'train_y.npy'))
    indices = y_train - 1
    # One-Hot-Encoding
    y_train = np.eye(8)[indices]
    x_train = norm(x_train)

    #print(x_train.shape)
    #print(y_train.shape)

    # set CNN model parameter
    input_shape = (x_train.shape[1]) # (365, 10)
    print(input_shape)
    lc_num = 8 # all LUCAS LC-Classes
    model = get_MLP_model(input_shape, lc_num)
    model.summary()

    # set train parameter
    lr = 1e-3
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    train_index = list(range(y_train.shape[0]))
    epochs = 100
    batch_size = 64
    iterations = int(y_train.shape[0]/batch_size)
    print(batch_size,iterations)
    random.shuffle(train_index)

    if not os.path.exists( os.path.join(args.working_directory, '3_trained_model', 'version' +str(model_number)) ):
        os.makedirs(os.path.join(args.working_directory, '3_trained_model', 'version' +str(model_number)) )

    with open(os.path.join(args.working_directory, '3_trained_model','version' +str(model_number),'performance.txt'), 'w') as file:
        file.write(f" Epoch ; CCE ; LR \n")

    #===========================================
    #================ TRAINING =================
    #===========================================
    # about 1 minute per Epoch
    for e in range(epochs):
        loss_train = 0
        for i in tqdm(range(iterations)):
            x_batch = x_train[train_index[i*batch_size: i*batch_size + batch_size]]
            y_batch = y_train[train_index[i*batch_size: i*batch_size + batch_size]]
            loss_train += train_step(x_batch, y_batch, model, opt)
        loss_train = loss_train / iterations # mean loss value
        loss_train = loss_train.numpy()
        print('Epoch: ', e)
        print("Mean loss (CCE):", np.mean(loss_train), "Std:", np.std(loss_train))
        with open(os.path.join(args.working_directory, '3_trained_model','version' +str(model_number),'performance.txt'), 'a') as file:
            file.write(f"{e};{np.mean(loss_train)};{lr}\n")
        random.shuffle(train_index)
        if e % 100 == 0:
            print('Save model at epoch: ', e)
            model_path = os.path.join(args.working_directory, '3_trained_model','version' +str(model_number), 'saved_model'+ str(model_number)+'_epoch'+ str(e)+'.keras')
            tf.keras.models.save_model(model, model_path)
            print('Model is saved at ', model_path)
    if True:
        #model_path = os.path.join(args.working_directory, '3_trained_model','version' +str(model_number))
        model_path = os.path.join(args.working_directory, '3_trained_model','version' +str(model_number), 'saved_model'+ str(model_number)+ '.keras')
        tf.keras.models.save_model(model, model_path)
        print('Model is saved at ', model_path)

if __name__ == '__main__':
    train(5)