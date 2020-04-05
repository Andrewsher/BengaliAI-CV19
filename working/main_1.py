import os
import argparse
parser = argparse.ArgumentParser(description='gpu')
parser.add_argument('-g', '--gpu', default='0', type=str)
parser.add_argument('-o', '--output-file', default='model.h5', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.makedirs('outputs', exist_ok=True)

import pandas as pd
import math
import cv2
import numpy as np
import keras

label_df = pd.read_csv('../input/bengaliai-cv19/train.csv')
label_df.head()
print(label_df.shape)

data0 = pd.read_feather('../input/feather-generation/train_data_00_l.feather')
data1 = pd.read_feather('../input/feather-generation/train_data_11_l.feather')
data2 = pd.read_feather('../input/feather-generation/train_data_22_l.feather')
data3 = pd.read_feather('../input/feather-generation/train_data_33_l.feather')
data_df = pd.concat([data0,data1,data2,data3],ignore_index=True)
del data0,data1,data2,data3

print(data_df.shape)


from keras.preprocessing.image import ImageDataGenerator as ImageDataGenerator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_df, label_df, is_train=False, batch_size=128, shuffle=False):
        self.data = data_df.iloc[:, 1:].values
        self.label = label_df
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = [i for i in range(len(self.label))]
        self.current_index = 0
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            # brightness_range=(0.5, 2),
            shear_range=0.3,
            zoom_range=0.3,
        )
        self.cutout = np.zeros((1, 16, 16, 1))

    def __len__(self):
        return len(self.label) // self.batch_size

    def __getitem__(self, item):
        batch_data = []
        batch_label_1, batch_label_2, batch_label_3 = [], [], []
        for i in range(self.batch_size):
            idx = self.indices[self.current_index]
            current_data = self.data[idx, :].reshape(128, 128, 1).astype(np.float) / 255.0
            batch_data.append(current_data)
            batch_label_1.append(self.label.grapheme_root.values[idx])
            batch_label_2.append(self.label.vowel_diacritic.values[idx])
            batch_label_3.append(self.label.consonant_diacritic.values[idx])
            self.current_index = (self.current_index + 1) % len(self.indices)

        batch_label_1 = keras.utils.to_categorical(batch_label_1, num_classes=168)
        batch_label_2 = keras.utils.to_categorical(batch_label_2, 11)
        batch_label_3 = keras.utils.to_categorical(batch_label_3, 7)

        # print(batch_label_1.shape, batch_label_2.shape, batch_label_3.shape)

        if self.is_train:
            x, y1, y2, y3 = self.data_aug(np.array(batch_data), np.array(batch_label_1), np.array(batch_label_2), np.array(batch_label_3))
            return x, [y1, y2, y3]
        else:
            return np.array(batch_data), \
               [np.array(batch_label_1), np.array(batch_label_2), np.array(batch_label_3)]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def rand_bbox(self, size, lam):
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def data_aug(self, x, y1, y2, y3, alpha=0.2):
        # data aug
        for i in range(len(x)):
            x[i, ...] = self.datagen.random_transform(x[i, ...])

        # label smooth
        label_smooth = np.random.uniform(0.15, 0.25, len(x))
        for i in range(len(x)):
            y1[i][y1[i] > 0.5] = 1.0 - label_smooth[i]
            y1[i][y1[i] < 0.5] = label_smooth[i] / 168.
            y2[i][y2[i] > 0.5] = 1.0 - label_smooth[i]
            y2[i][y2[i] < 0.5] = label_smooth[i] / 11.
            y3[i][y3[i] > 0.5] = 1.0 - label_smooth[i]
            y3[i][y3[i] < 0.5] = label_smooth[i] / 7.

        # mixup
        if np.random.random() > 0.5:
            weight = np.random.beta(alpha, alpha, len(x))
            x_weight = weight.reshape((self.batch_size, 1, 1, 1))
            y_weight = weight.reshape((self.batch_size, 1))
            index = np.random.permutation(self.batch_size)
            x = x * x_weight + x[index] * (1 - x_weight)
            y1 = y1 * y_weight + y1[index] * (1 - y_weight)
            y2 = y2 * y_weight + y2[index] * (1 - y_weight)
            y3 = y3 * y_weight + y3[index] * (1 - y_weight)

        # cutmix
        if np.random.random() > 0.5:
            indices = np.random.permutation(len(x))
            lam = np.random.beta(alpha, alpha)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.shape, lam)
            x[:, bbx1:bbx2, bby1:bby2, :] = x[indices, bbx1:bbx2, bby1:bby2, :]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[1] * x.shape[2]))
            y1 = y1 * lam + y1[indices] * (1 - lam)
            y2 = y2 * lam + y2[indices] * (1 - lam)
            y3 = y3 * lam + y3[indices] * (1 - lam)

        # cutout
        if np.random.random() > 0.5:
            for i in range(len(x)):
                left = np.random.randint(0, 96)
                top = np.random.randint(0, 96)
                x[i, left:left+16, top:top+16, :] = self.cutout

        return x, y1, y2, y3



'''
SnapShot
'''
import keras.callbacks as callbacks
from keras.callbacks import Callback


class SnapshotCallbackBuilder:
    """Callback builder for snapshot ensemble training of a model.
    Creates a list of callbacks, which are provided when training a model
    so as to save the model weights at certain epochs, and then sharply
    increase the learning rate.
    """

    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        """
        Initialize a snapshot callback builder.
        # Arguments:
            nb_epochs: total number of epochs that the model will be trained for.
            nb_snapshots: number of times the weights of the model will be saved.
            init_lr: initial learning rate
        """
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self):
        """
        Creates a list of callbacks that can be used during training to create a
        snapshot ensemble of the model.
        Args:
            model_prefix: prefix for the filename of the weights.
        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,
                 SnapshotModelCheckpoint] which can be provided to the 'fit' function
        """
        # if not os.path.exists(self.weights_path):
        #     os.makedirs(self.weights_path)

        # callback_list = [
        #     callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule),
        #     SnapshotModelCheckpoint(self.T, self.M, os.path.join(self.weights_path, model_prefix), fold)]

        return callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return max(float(self.alpha_zero / 2 * cos_out), 1e-8)
        # return float(self.alpha_zero / 2 * cos_out)

import keras
from keras.layers import Dense, Flatten, BatchNormalization, ReLU, Dropout, Input, Concatenate, GlobalAveragePooling2D, Conv2D
from keras.models import Model
from keras_applications.densenet import DenseNet201, DenseNet169, DenseNet121
from keras_applications.resnext import ResNeXt101, ResNeXt50
from keras_applications.inception_v3 import InceptionV3
from keras_applications.inception_resnet_v2 import InceptionResNetV2
from keras_applications.resnet_v2 import ResNet152V2, ResNet101V2, ResNet50V2
from keras_applications.mobilenet_v2 import MobileNetV2
from classification_models.models.resnet import ResNet18
from classification_models.models.senet import SEResNeXt50, SEResNeXt101, SEResNet50, SEResNet101, SEResNet152
from keras_applications.vgg19 import VGG19
from efficientnet.keras import EfficientNetB6
from efficientnet.keras import EfficientNetB4


def build_model():
    encoder = SEResNet101(weights='imagenet', input_shape=(128, 128, 3), include_top=False, backend=keras.backend,
                         layers=keras.layers, models=keras.models, utils=keras.utils)
    input_tensor = Input(shape=(128, 128, 1))
    # x = Conv2D(3, (3, 3), padding='same', name='pre_conv')(input_tensor)
    x = Concatenate()([input_tensor, input_tensor, input_tensor])

    x = encoder(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x_1 = Dense(168, activation='softmax', name='grapheme')(x)
    x_2 = Dense(11, activation='softmax', name='vowel')(x)
    x_3 = Dense(7, activation='softmax', name='consonant')(x)

    model = Model(inputs=input_tensor, outputs=[x_1, x_2, x_3])
    return model


from keras import optimizers
model = build_model()
model.compile(optimizers.Adam(lr=0.0001),
              metrics=[keras.metrics.Recall(name='recall')],
              loss='categorical_crossentropy')
# model.summary()



from sklearn.model_selection import train_test_split
train_label_df, val_label_df = train_test_split(label_df,test_size=0.20, random_state=42,shuffle=True) ## Split Labels
train_data_df, val_data_df = train_test_split(data_df,test_size=0.20, random_state=42,shuffle =True) ## split data
del data_df

batch_size = 64


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_delta=0.2, patience=5, verbose=1, mode='min', min_lr=1e-12)

model.fit_generator(DataGenerator(train_data_df, train_label_df, is_train=True, batch_size=batch_size, shuffle=True),
                    validation_data=DataGenerator(val_data_df, val_label_df, is_train=False, batch_size=batch_size, shuffle=False),
                    epochs=100,
                    initial_epoch=0,
                    verbose=1,
                    callbacks=[reduce_lr])

model.save(os.path.join('outputs', args.output_file))
