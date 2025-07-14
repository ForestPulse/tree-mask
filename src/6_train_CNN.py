import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the working directory", default= "/data/ahsoka/eocp/forestpulse/INTERNAL/tree_mask")
args = parser.parse_args()

def train(model_number):
    def norm(a):
        a_out = a/10000.
        return a_out
    
    def get_model(input_shape, lc_num, no_filters, kernel_size):

        print(input_shape)
        x_in = tf.keras.Input(shape=input_shape)
        # First Convolutional Layer
        x = tf.keras.layers.Conv1D(filters= no_filters, kernel_size=kernel_size)(x_in)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        # repeated Convolutional Layer
        for _ in range(2):
            x = tf.keras.layers.Conv1D(filters= no_filters, kernel_size=kernel_size)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        # Flatten und Dense Layer  
        x =  tf.keras.layers.Flatten()(x)
        x_out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        #x_out = tf.keras.layers.Dense(lc_num, activation="softmax")(x)

        model = tf.keras.Model(inputs = x_in, outputs = x_out)
        return model
    
    def get_loss(x, y, model, training=True):
        y_pred = model(x, training=training)
        loss = tf.keras.losses.BinaryCrossentropy()(y, y_pred)
        return loss
    
    @tf.function
    def train(x, y, model, opt):
        with tf.GradientTape() as tape:
            loss = get_loss(x, y, model, training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    print("trained model " +str(model_number))
    x_train= np.load(os.path.join(args.working_directory, '4_augmented_data', 'TE_augmented_x.npy'))
    y_train= np.load(os.path.join(args.working_directory, '4_augmented_data', 'TE_augmented_y.npy'))

    x_train = norm(x_train)
    print(x_train.shape)

    # set CNN model parameter
    input_shape = (x_train.shape[1], x_train.shape[2])
    print(input_shape)
    lc_num = 2 # Forest and Non-Forest
    no_filters = 64
    kernel_size = 16
    model = get_model(input_shape, lc_num, no_filters, kernel_size)
    model.summary()

    # set train parameter
    lr = 1e-3
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    train_index = list(range(y_train.shape[0]))
    epochs = 5
    batch_size = 64
    iterations = int(y_train.shape[0]/batch_size)
    print(batch_size,iterations)
    random.shuffle(train_index)

    if not os.path.exists( os.path.join(args.working_directory, '5_trained_model', 'version' +str(model_number)) ):
        os.makedirs(os.path.join(args.working_directory, '5_trained_model', 'version' +str(model_number)) )

    with open(os.path.join(args.working_directory, '5_trained_model','version' +str(model_number),'performance.txt'), 'w') as file:
        file.write(f" Epoch ; MAE ; LR \n")

    #===========================================
    #================ TRAINING =================
    #===========================================
    for e in range(epochs):
        loss_train = 0
        for i in tqdm(range(iterations)):
            x_batch = x_train[train_index[i*batch_size: i*batch_size + batch_size]]
            y_batch = y_train[train_index[i*batch_size: i*batch_size + batch_size]]
            loss_train += train(x_batch, y_batch, model, opt)
        loss_train = loss_train / iterations
        loss_train = loss_train.numpy()
        print('Epoch: ', e)
        print('BCE: ', loss_train)
        with open(os.path.join(args.working_directory, '5_trained_model','version' +str(model_number),'performance.txt'), 'a') as file:
            file.write(f"{e};{loss_train};{lr}\n")
        random.shuffle(train_index)
        #lr *= params['LEARNING_RATE_DECAY']
        #opt.learning_raye = lr
    if True:
        #model_path = os.path.join(args.working_directory, '5_trained_model','version' +str(model_number))
        model_path = os.path.join(args.working_directory, '5_trained_model','version' +str(model_number), 'saved_model'+ str(model_number)+ '.keras')
        tf.keras.models.save_model(model, model_path)
        print('Model is saved at ', model_path)

if __name__ == '__main__':
    train(1)