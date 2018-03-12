import logging
import glob
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
# from keras.optimizers import RMSprop
from keras import optimizers as opt
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras import regularizers
import os
import time

## to create a list of dependencies for using the repo
## do : pip freeze > requirements.txt

np.random.seed(1337)
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
# from models.modified_vgg_16_model import modified_vgg_16l
# from models.google_net import modified_googlenet

"""
64 * 64 for VGG16
224 * 224 for GoogleNet
"""
WIDTH = 64
HEIGHT = 64

create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]
class Logger(object):
      def __init__(self, experiment_name = '', folder='./results' ):
            """
            Saves experimental metrics for use later.
            :param experiment_name: name of the experiment
            :param folder: location to save data
            """

            self.train_acc = []
            self.valid_acc = []
            self.train_loss = []
            self.valid_loss = []
            self.test_acc = []

            self.save_folder = os.path.join(folder, experiment_name, time.strftime('%y-%m-%d-%H-%M-%s'))
            # self.save_folder = os.path.join(folder, experiment_name, lambda_values, time.strftime('%y-%m-%d-%H-%M-%s'))
            create_folder(self.save_folder)

      def record_data(self, train_acc, valid_acc, train_loss, valid_loss, accuracy_score):
            self.train_acc = train_acc
            self.valid_acc = valid_acc
            self.train_loss = train_loss
            self.valid_loss = valid_loss
            self.test_acc = accuracy_score

      def save(self):
            np.save(os.path.join(self.save_folder, "train_acc.npy"), self.train_acc)
            np.save(os.path.join(self.save_folder, "valid_acc.npy"), self.valid_acc)
            np.save(os.path.join(self.save_folder, "test_acc.npy"), self.test_acc)
            np.save(os.path.join(self.save_folder, "train_loss.npy"), self.train_loss)
            np.save(os.path.join(self.save_folder, "valid_loss.npy"), self.valid_loss)



      def save_args(self, args):
            """
            Save the command line arguments
            """
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs()), f)




# def create_logger():
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.DEBUG)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)
#     return logger


def convert_image_to_data(image, WIDTH, HEIGHT):
    image_resized = Image.open(image).resize((WIDTH, HEIGHT))
    image_array = np.array(image_resized).T
    return image_array


def create_train_test_data(WIDTH, HEIGHT):

    # cat_files = glob.glob("data/train/cat*")
    # dog_files = glob.glob("data/train/dog*")

    """
    For simple testing purposes
    """
    cat_files = glob.glob("data_sample/train/cat*")
    dog_files = glob.glob("data_sample/train/dog*")


    # Restrict cat and dog files here for testing
    cat_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in cat_files]
    dog_list = [convert_image_to_data(i, WIDTH, HEIGHT) for i in dog_files]

    y_cat = np.zeros(len(cat_list))
    y_dog = np.ones(len(dog_list))

    X = np.concatenate([cat_list, dog_list])
    X = np.concatenate([cat_list, dog_list])
    y = np.concatenate([y_cat, y_dog])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test


def train_model(model, X_data_train, y_target_train, early_stopping):
    """
    :param model: compiled model
    :param X_data_train: 3d array
    :param y_target_train: 1d array
    :param flag: true if googlnet (output expect 3d array) else false if 1d array for output
    :return: fitted model
    """

    if early_stopping == "early":
        early_stopping = EarlyStopping(monitor="loss", patience=3)
        ## nb_epoch = 20
        data = model.fit(X_data_train, y_target_train, batch_size=32, nb_epoch=2, validation_split=0.2, callbacks=[early_stopping])
    elif early_stopping == "no_early":
        data = model.fit(X_data_train, y_target_train, batch_size=32, nb_epoch=2, validation_split=0.2)

        
    train_acc = data.history['acc']
    valid_acc = data.history['val_acc']
    train_loss = data.history['loss']
    valid_loss = data.history['val_loss']


    return model, train_acc, valid_acc, train_loss, valid_loss


def evaluate_model(model, X_data_test, y_target_test):
    y_test_predict = model.predict_classes(X_data_test)

    return accuracy_score(y_target_test, y_test_predict)


def modified_vgg_16l(width, height, optimizer, normalization, regularizer):
    model = Sequential()
    # input: 64x64 images with 3 channels -> (3, 64, 64) tensors.
    # this applies 64 convolution filters of size 3x3 each.
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, width, height)))

    if normalization == "Batch":
        model.add(BatchNormalization())
    elif normalization == "Layer":
        model.add(BatchNormalization(mode=1))


    model.add(Activation('relu'))

    if regularizer == "Dropout":
        model.add(Dropout(0.1))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))

    if normalization == "Batch":
        model.add(BatchNormalization())
    elif normalization == "Layer":
        model.add(BatchNormalization(mode=1))

    model.add(Activation('relu'))

    if regularizer == "Dropout":
        model.add(Dropout(0.1))


    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    if regularizer == "Dropout":
        model.add(Dropout(0.1))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, 3, 3, border_mode='same'))

    if normalization == "Batch":
        model.add(BatchNormalization())
    elif normalization == "Layer":
        model.add(BatchNormalization(mode=1))

    model.add(Activation('relu'))
    if regularizer == "Dropout":
        model.add(Dropout(0.1))

    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    if regularizer == "Dropout":
        model.add(Dropout(0.1))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256))

    if normalization == "Batch":
        model.add(BatchNormalization())
    elif normalization == "Layer":
        model.add(BatchNormalization(mode=1))

    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    if optimizer == "RMSProp":
        model.compile(loss='binary_crossentropy', optimizer=opt.RMSprop(lr=1e-4), metrics=['accuracy'])
    elif optimizer == "ADAM":
        model.compile(loss='binary_crossentropy', optimizer=opt.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])
    elif optimizer == "SGD":
        model.compile(loss='binary_crossentropy', optimizer=opt.SGD(lr=0.001), metrics=['accuracy'])
    elif optimizer == "Adagrad":
        model.compile(loss='binary_crossentropy', optimizer=opt.Adagrad(lr=0.01), metrics=['accuracy'])

    return model



if __name__ == "__main__":
    print ("Create train and test dataset")
    parser = argparse.ArgumentParser()

    ## optimizers and normalizer to be added when defining the keras model
    parser.add_argument("--optimizer", default="ADAM", help="ADAM, RMSProp, SGD, Adagrad")                  # Optimizers
    parser.add_argument("--norm", default="Batch", help="Batch Norm, noflag = No Batch Norm")              # Normalizer
    parser.add_argument("--regularizer", default="Dropout", help="Dropout, noflag = No Dropout on All Layers, just final layer")              # Regularizer
    parser.add_argument("--early", default="early", help="early, no_early")              # Regularizer
    args = parser.parse_args()

    optimizer = args.optimizer
    normalization = args.norm
    regularizer = args.regularizer
    early_stopping = args.early

    logger = Logger(experiment_name = optimizer + '_' + normalization + '_' + regularizer + '_' + early_stopping, folder = './results')
    # logger = Logger(optimizer=optimizer, norm=normalization, regularizer=regularizer, early=early_stopping, folder='./results' )

    X_train, X_test, y_train, y_test = create_train_test_data(WIDTH, HEIGHT)
    print("Shape for X_train: " + str(X_train.shape) + " Shape for y_train: " + str(y_train.shape))


    print("Create the model.")
    model = modified_vgg_16l(WIDTH, HEIGHT, optimizer, normalization, regularizer)

    print("Train the model")
    # flag=False if no GoogleNet
    trained_model, train_acc, valid_acc, train_loss, valid_loss  = train_model(model, X_train, y_train, early_stopping)


    print ("Evaluate the model")
    accuracy_score = evaluate_model(trained_model, X_test, y_test)
    print("The accurarcy is " + str(accuracy_score))
    print("Save model")

    logger.record_data(train_acc, valid_acc, train_loss, valid_loss, accuracy_score)
    logger.save()

    # np.save('./results/' + 'train_acc', np.asarray(train_acc))
    # np.save('./results/' + 'valid_acc', np.asarray(valid_acc))
    # np.save('./results/' + 'train_loss', np.asarray(train_loss))
    # np.save('./results/' + 'valid_loss', np.asarray(valid_loss))
    # np.save('./results/' + 'train_acc', np.asarray(train_acc))



