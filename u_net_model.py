'''Running this file creates and train a neural network following the U-Net Architecture. Parameters are optimized but can be changes to suit computational power'''

'''General Imports'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from itertools import chain
from tensorflow.keras import Input
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from Datasets.mask_to_submission import masks_to_submission

'''Global variables '''

'''Directories'''
OVERALL_DIRECTORY = ''  # Overall path of the files
DATA_DIRECTORY = OVERALL_DIRECTORY + 'Datasets/training/'  # Training data path
TRAIN_DATA_FILENAME = DATA_DIRECTORY + 'images/'  # Training RGB images path
TRAIN_LABELS_FILENAME = DATA_DIRECTORY + \
    'groundtruth/'  # Training groundtruth images path

'''Parameters'''
ALPHA = 0.1  # Alpha value for Leaky ReLU


def lrelu(x): return tf.keras.layers.LeakyReLU(alpha=ALPHA)(
    x)  # Leaky ReLU lambda. To insert in ACTIVATION if used


# Set to True to continuously generate images (computation heavy). False will generate an initial batch of images that the model will use to train
GENERATOR = True
SEED = 1  # Set to None for random seed.
BATCH_SIZE = 18
NUM_EPOCHS = 500
DROPOUT = 0.25
ACTIVATION = 'relu'  # Can change to lrelu
ROTATION = 180  # Range of rotation of the generated images
NB_FILTERS = 32  # Initial convolution depth
VAL_SPLIT = 0.2  # Percentage used for validation. Only used when generator = False

STEPS_PER_EPOCH = 100
KERNEL_SIZE = 3  # Convolution layers kernel size
# Number of images used for training. Only used when generator = False
TRAINING_SIZE = STEPS_PER_EPOCH * BATCH_SIZE
# Number of generated images. Only used when generator = False. Can generate memory error
GEN_SIZE = int(TRAINING_SIZE / (1 - VAL_SPLIT))
# Number of images used for validation. Only used when generator = False
VALIDATION_SIZE = GEN_SIZE - TRAINING_SIZE

'''
Load images contain in a folder and transform load images into arrays.
PARAMETERS:
    - images_path: path where to load images
    - groundtruth: boolean variable to know if the images to load are groundtruth images.
RETURN:
    - Array containing every array representation of the images contained in images_path.
'''


def extract_images(images_path, groundtruth=False):
    imgs = []
    for root, _, files in os.walk(images_path, topdown=True):
        for fn in sorted(files):
            image_filename = os.path.join(root, fn)
            if os.path.isfile(image_filename):
                print('Loading ' + image_filename)
                if (groundtruth == True):
                    img = load_img(
                        image_filename, color_mode="grayscale", target_size=(400, 400))
                else:
                    img = load_img(image_filename, target_size=(400, 400))
                imgs.append(img_to_array(img) / 255)
            else:
                print('File ' + image_filename + ' does not exist')
    return np.array(imgs)


data = extract_images(images_path=TRAIN_DATA_FILENAME)
labels = extract_images(images_path=TRAIN_LABELS_FILENAME, groundtruth=True)


'''
Use in the case of generator = True
Combine keras generators to output a single generator. Needed to use as argument in fit_generator()
PARAMETERS:
    - gen_x: keras generator generating the data
    - gen_y: keras generator generating the labels
YIELD:
    a single generator which generate the data and the labels.
'''


def combine_generator(gen_x, gen_y):
    while True:
        yield(next(gen_x), next(gen_y))


'''
Use in the case of generator = False
Generate a list from a keras generator.
PARAMETER:
    - gen: keras generator
RETURN:
    Array containing GEN_SIZE number of images from the generator gen.
'''


def generate_list(gen):
    list = []
    print('Loading generation ...')
    for i in range(GEN_SIZE):
        list.append(next(gen))
    return np.squeeze(np.array(list))


'''If generator is True, then we create an image generator for the data and for
the labels and we combine them. Otherwise, we use the same generator to create
 a list of GEN_SIZE elements'''
if (GENERATOR):
    generator = ImageDataGenerator(
        rotation_range=ROTATION, horizontal_flip=True, fill_mode='reflect')
    train_x_generator = generator.flow(
        data, batch_size=BATCH_SIZE, seed=SEED)
    train_y_generator = generator.flow(
        labels, batch_size=BATCH_SIZE, seed=SEED)

    train_generator = combine_generator(train_x_generator, train_y_generator)

else:
    generator = ImageDataGenerator(
        rotation_range=ROTATION, horizontal_flip=True, fill_mode='reflect')
    train_x = generator.flow(
        data, batch_size=1, seed=SEED)
    train_y = generator.flow(
        labels, batch_size=1, seed=SEED)

    x = generate_list(train_x)
    y = np.expand_dims(generate_list(train_y), axis=3)

    indices = np.arange(len(x))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]


'''
Plot the training F1 score as well as the F1 score on the validation set for each epoch
PARAMETER:
    history: holds a record of the loss values and metric values during training and validation.
'''


def f1_plot(history):
    print('\nhistory dict:', history.history)
    history_df = pd.DataFrame.from_dict(history.history)
    history_df.to_pickle(OVERALL_DIRECTORY + 'history.pkl')

    print('\n plotting F1 score ...')

    train_precision = np.array(history.history['precision'])
    train_recall = np.array(history.history['recall'])
    train_f1_score = 2 * (train_precision * train_recall) / \
        (train_precision + train_recall)

    validation_precision = np.array(history.history['val_precision'])
    validation_recall = np.array(history.history['val_recall'])
    validation_f1_score = 2 * \
        (validation_precision * validation_recall) / \
        (validation_precision + validation_recall)

    # Plot training & validation accuracy values
    plt.plot(train_f1_score)
    plt.plot(validation_f1_score)
    plt.title('Model F1 score')
    plt.ylabel('F1 score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(OVERALL_DIRECTORY + 'f1_score.png')


"""
U-Net model architecture
PARAMETERS:
    - data: input data
    - activation_function: activation function used for the model.
    - dropout: number from 0 to 1 indicating the amount of dropout.
    - nb_filters: initial depth of the convolutions.
RETURN:
    model: keras instance of the U-Net model.
"""


def model(data, activation_function=ACTIVATION, dropout=DROPOUT, nb_filters=NB_FILTERS):
    '''
    Helper method for repetitive convolution and ReLU operations.
    '''
    def CRCR(data, nb_filters, kernel_size):
        cr = Conv2D(filters=nb_filters, kernel_size=(kernel_size, kernel_size),
                    kernel_initializer="random_uniform", padding="same", activation=activation_function)(data)
        cr = BatchNormalization()(cr)

        cr = Conv2D(filters=nb_filters, kernel_size=(kernel_size, kernel_size),
                    kernel_initializer="random_uniform", padding="same", activation=activation_function)(cr)
        cr = BatchNormalization()(cr)
        return cr

    '''Contraction'''
    cr1 = CRCR(data, nb_filters, KERNEL_SIZE)
    pool1 = MaxPool2D()(cr1)
    drop1 = Dropout(dropout)(pool1)

    cr2 = CRCR(drop1, 2 * nb_filters, KERNEL_SIZE)
    pool2 = MaxPool2D()(cr2)
    drop2 = Dropout(dropout)(pool2)

    cr3 = CRCR(drop2, 2 * 2 * nb_filters, KERNEL_SIZE)
    pool3 = MaxPool2D()(cr3)
    drop3 = Dropout(dropout)(pool3)

    cr4 = CRCR(drop3, 2 * 2 * 2 * nb_filters, KERNEL_SIZE)
    pool4 = MaxPool2D()(cr4)
    drop4 = Dropout(dropout)(pool4)

    cr5 = CRCR(drop4, 2 * 2 * 2 * 2 * nb_filters, KERNEL_SIZE)

    '''Expansion'''
    upconv4 = Conv2DTranspose(
        2 * 2 * 2 * nb_filters, (KERNEL_SIZE, KERNEL_SIZE), strides=(2, 2), padding='same')(cr5)
    concat4 = concatenate([upconv4, cr4])
    upcr4 = CRCR(concat4, 2 * 2 * 2 * nb_filters, KERNEL_SIZE)

    upconv3 = Conv2DTranspose(
        2 * 2 * nb_filters, (KERNEL_SIZE, KERNEL_SIZE), strides=(2, 2), padding='same')(upcr4)
    concat3 = concatenate([upconv3, cr3])
    upcr3 = CRCR(concat3, 2 * 2 * nb_filters, KERNEL_SIZE)

    upconv2 = Conv2DTranspose(
        2 * nb_filters, (KERNEL_SIZE, KERNEL_SIZE), strides=(2, 2), padding='same')(upcr3)
    concat2 = concatenate([upconv2, cr2])
    upcr2 = CRCR(concat2, 2 * nb_filters, KERNEL_SIZE)

    upconv1 = Conv2DTranspose(
        nb_filters, (KERNEL_SIZE, KERNEL_SIZE), strides=(2, 2), padding='same')(upcr2)
    concat1 = concatenate([upconv1, cr1])
    upcr1 = CRCR(concat1, nb_filters, KERNEL_SIZE)

    result = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(upcr1)
    model = tf.keras.Model(inputs=data, outputs=result)

    return model


print("creating model ...\n")
data = Input((400, 400, 3))
model = model(data)
# Specify the training configuration (optimizer, loss, metrics)
model.compile(optimizer=tf.keras.optimizers.Adam(),  # Optimizer
              # Loss function to minimize
              loss=tf.keras.losses.BinaryCrossentropy(
                  from_logits=True),
              # List of metrics to monitor
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Show the model architecture
model.summary()

'''Train the model by slicing the data into STEPS_PER_EPOCH batches
of size BATCH_SIZE, and repeatedly iterating over
the entire dataset for a given number of epochs (NUM_EPOCHS).
Can use a generator to continuously generate new images (computation heavy).'''
print('# Fit model on training data')
if (GENERATOR):
    history = model.fit_generator(train_generator,
                                  epochs=NUM_EPOCHS,
                                  steps_per_epoch=STEPS_PER_EPOCH)

else:
    history = model.fit(x, y, batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS, validation_split=VAL_SPLIT)
    f1_plot(history)

# Save the variables to disk.
save_path = OVERALL_DIRECTORY + "unet_model.h5"
model.save(save_path)
print("Model saved in path: %s" % save_path)
