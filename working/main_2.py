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
        self.cutout = np.zeros((1, 32, 32, 1))

    def __len__(self):
        return int(np.ceil(len(self.label) / float(self.batch_size)))

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
        label_smooth = np.random.uniform(0.0, 0.05, len(x))
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
            # x_a, x_b = x, x[index]
            # x = x_a * x_weight + x_b * (1 - x_weight)
            x = x * x_weight + x[index] * (1 - x_weight)
            y1 = y1 * y_weight + y1[index] * (1 - y_weight)
            y2 = y2 * y_weight + y2[index] * (1 - y_weight)
            y3 = y3 * y_weight + y3[index] * (1 - y_weight)

        # cutmix
        elif np.random.random() > 0.66:
            indices = np.random.permutation(len(x))
            lam = np.random.beta(alpha, alpha)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.shape, lam)
            x[:, bbx1:bbx2, bby1:bby2, :] = x[indices, bbx1:bbx2, bby1:bby2, :]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[1] * x.shape[2]))
            y1 = y1 * lam + y1[indices] * (1 - lam)
            y2 = y2 * lam + y2[indices] * (1 - lam)
            y3 = y3 * lam + y3[indices] * (1 - lam)

        return x, y1, y2, y3



import keras
from keras.layers import Dense, Flatten, BatchNormalization, ReLU, Dropout, Input, Concatenate, GlobalAveragePooling2D, Conv2D, UpSampling2D
from keras.models import Model
from keras_applications.densenet import DenseNet201, DenseNet169, DenseNet121
from keras_applications.resnext import ResNeXt101, ResNeXt50
from keras_applications.inception_v3 import InceptionV3
from keras_applications.resnet_v2 import ResNet152V2, ResNet101V2, ResNet50V2
from keras_applications.mobilenet_v2 import MobileNetV2
from keras_applications.vgg19 import VGG19
from keras_applications.vgg16 import VGG16
from classification_models.models.senet import SEResNeXt50, SEResNeXt101, SEResNet152, SEResNet50, SEResNet101
from classification_models.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import efficientnet.keras as efn


batch_size = 64
def build_model():
    encoder = SEResNeXt50(weights='imagenet', input_shape=(128, 128, 3), include_top=False, backend=keras.backend,
                         layers=keras.layers, models=keras.models, utils=keras.utils)
    encoder.trainable = True
    input_tensor = Input(shape=(128, 128, 1))
    # x = Conv2D(3, (3, 3), padding='same', name='pre_conv')(input_tensor)
    x = Concatenate()([input_tensor, input_tensor, input_tensor])

    x = encoder(x)

    x = encoder.output
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(rate=0.2)(x)


    x_1 = Dense(168, activation='softmax', name='grapheme')(x)
    x_2 = Dense(11, activation='softmax', name='vowel')(x)
    x_3 = Dense(7, activation='softmax', name='consonant')(x)

    model = Model(inputs=input_tensor, outputs=[x_1, x_2, x_3])
    return model

import keras.backend as K
class RAdam(keras.optimizers.Optimizer):
    """RAdam optimizer.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay for each param.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        total_steps: int >= 0. Total number of training steps. Enable warmup by setting a positive value.
        warmup_proportion: 0 < warmup_proportion < 1. The proportion of increasing steps.
        min_lr: float >= 0. Minimum learning rate after warmup.
    # References
        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0., amsgrad=False,
                 total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):
        learning_rate = kwargs.pop('lr', learning_rate)
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.total_steps = K.variable(total_steps, name='total_steps')
            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')
            self.min_lr = K.variable(min_lr, name='min_lr')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.initial_total_steps = total_steps
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        if self.initial_total_steps > 0:
            warmup_steps = self.total_steps * self.warmup_proportion
            decay_steps = K.maximum(self.total_steps - warmup_steps, 1)
            decay_rate = (self.min_lr - lr) / decay_steps
            lr = K.switch(
                t <= warmup_steps,
                lr * (t / warmup_steps),
                lr + decay_rate * K.minimum(t - warmup_steps, decay_steps),
            )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]

        self.weights = [self.iterations] + ms + vs + vhats

        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0
        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            m_corr_t = m_t / (1.0 - beta_1_t)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t))
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t))

            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                         (sma_t - 2.0) / (sma_inf - 2.0) *
                         sma_inf / sma_t)

            p_t = K.switch(sma_t >= 5, r_t * m_corr_t / (v_corr_t + self.epsilon), m_corr_t)

            if self.initial_weight_decay > 0:
                p_t += self.weight_decay * p

            p_t = p - lr * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, learning_rate):
        self.learning_rate = learning_rate

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.learning_rate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': float(K.get_value(self.total_steps)),
            'warmup_proportion': float(K.get_value(self.warmup_proportion)),
            'min_lr': float(K.get_value(self.min_lr)),
        }
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from keras import optimizers
# keras.metrics.Recall
model = build_model()
model.compile(optimizers.Adam(lr=0.0001),
              metrics=[keras.metrics.Recall(name='recall')],
              loss='categorical_crossentropy')
# model.summary()



from sklearn.model_selection import train_test_split
train_label_df, val_label_df = train_test_split(label_df,test_size=0.20, random_state=1942,shuffle=True) ## Split Labels
train_data_df, val_data_df = train_test_split(data_df,test_size=0.20, random_state=1942,shuffle =True) ## split data


def get_lr(epoch):
    if epoch < 10:
        return 1e-4
    elif epoch < 30:
        return 1e-5
    elif epoch < 40:
        return 1e-6
    elif epoch < 60:
        return 1e-7
    else:
        return 1e-8
lr_strategy = keras.callbacks.LearningRateScheduler(get_lr)

model.fit_generator(DataGenerator(data_df, label_df, is_train=True, batch_size=batch_size, shuffle=True),
                    validation_data=DataGenerator(val_data_df, val_label_df, is_train=False, batch_size=batch_size, shuffle=False),
                    epochs=100,
                    initial_epoch=0,
                    verbose=1,
                    callbacks=[lr_strategy])

model.save(os.path.join('outputs', args.output_file))