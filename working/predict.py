import os
import argparse
parser = argparse.ArgumentParser(description='gpu')
parser.add_argument('-g', '--gpu', default='0', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# 从parquet文件进行预测
'''
test
'''

model_list = ['outputs/model_1.h5']
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import keras
HEIGHT = 137
WIDTH = 236
SIZE = 128


import keras

# prepare data
HEIGHT = 137
WIDTH = 236
size = (128, 128)

import cv2
from tqdm import tqdm

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=size, pad=16):
    #crop a box around pixels large than the threshold
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,size)

def Resize(df,size=size):
    resized = {}
    df = df.set_index('image_id')
    for i in tqdm(range(df.shape[0])):
       # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))
        image0 = 255 - df.loc[df.index[i]].values.reshape(137,236).astype(np.uint8)
    #normalize each image by its max val
        img = (image0*(255.0/image0.max())).astype(np.uint8)
        image = crop_resize(img)
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized

from keras.preprocessing.image import ImageDataGenerator as ImageDataGenerator
# def process_testdata(data, is_aug=False):
#     datagen = ImageDataGenerator(
# #             rotation_range=20,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
# #             brightness_range=(0.5, 2),
#             shear_range=0.3,
#             zoom_range=0.3,
#         )
#     data = Resize(data)
#     data = data.iloc[:, 1:].values
#     new_data = []
#     for idx in range(len(data)):
#         if is_aug:
#             new_data.append(datagen.random_transform(data[idx, :].reshape(128, 128, 1).astype(np.float) / 255.0))
#         else:
#             new_data.append(data[idx, :].reshape(128, 128, 1).astype(np.float) / 255.0)
#     del data
#     return np.array(new_data)



class TestDataGenerator(keras.utils.Sequence):
    def __init__(self, data_df, batch_size=1, is_aug=False):
        self.data = data_df.iloc[:, 1:].values
        self.batch_size = batch_size
        self.indices = [i for i in range(len(self.data))]
        self.current_index = 0
        self.datagen = ImageDataGenerator(
#             rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
#             brightness_range=(0.5, 2),
            shear_range=0.2,
            zoom_range=0.2,
        )
        self.is_aug = is_aug

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, item):
        batch_data = []
        for i in range(self.batch_size):
            idx = self.indices[self.current_index]
            current_data = self.data[idx, :].reshape(128, 128, 1).astype(np.float) / 255.0
            # current_data = (current_data.astype(np.float32) - 0.0692) / 0.2051
            if self.is_aug:
                current_data = self.datagen.random_transform(current_data)
            batch_data.append(current_data)
            self.current_index = (self.current_index + 1) % len(self.indices)
            if self.current_index == 0:
                return np.array(batch_data)

        return np.array(batch_data)

batch_size = 1
PATH = '../input/bengaliai-cv19/'
predictions = []

# TEST = ['../input/test-feather-generation/test_data_00_l.feather',
#         '../input/test-feather-generation/test_data_11_l.feather',
#         '../input/test-feather-generation/test_data_22_l.feather',
#         '../input/test-feather-generation/test_data_33_l.feather']
from keras.models import load_model

final_predictions=[]
# for fname in TEST:
for i in range(4):
    grapheme_prediction, vowel_prediction, consonant_prediction = [], [], []
    # data = pd.read_feather(fname)
    data = pd.read_parquet(PATH + f'test_image_data_{i}.parquet')
    data = Resize(data)
    for model_idx in range(len(model_list)):
        keras.backend.clear_session()
        model = load_model(model_list[model_idx], compile=False)
        # model.load_weights(weights_list[model_idx], by_name=True, skip_mismatch=True)
        # model.compile(optimizers.Adam(lr=0.0001), metrics=['accuracy'], loss='sparse_categorical_crossentropy')
        # predict original map
        current_prediction = model.predict_generator(TestDataGenerator(data_df=data, is_aug=False, batch_size=128),
                                                     verbose=1)
        # print(np.array(current_prediction).shape)
        grapheme_prediction.append(current_prediction[0])
        vowel_prediction.append(current_prediction[1])
        consonant_prediction.append(current_prediction[2])

        # predict TTA
        for _ in range(4):
            current_prediction = model.predict_generator(TestDataGenerator(data_df=data, is_aug=True, batch_size=128),
                                                         verbose=1)
            # print(np.array(current_prediction).shape)
            grapheme_prediction.append(current_prediction[0])
            vowel_prediction.append(current_prediction[1])
            consonant_prediction.append(current_prediction[2])

    # merge
    # grapheme_prediction = np.mean(grapheme_prediction, axis=0)
    # vowel_prediction = np.mean(vowel_prediction, axis=0)
    # consonant_prediction = np.mean(consonant_prediction, axis=0)

    [n_models, n_instances, _] = np.array(grapheme_prediction).shape

    new_grapheme_prediction = np.reshape(grapheme_prediction, [n_models, -1])
    new_vowel_prediction = np.reshape(vowel_prediction, [n_models, -1])
    new_consonant_prediction = np.reshape(consonant_prediction, [n_models, -1])

    grapheme_prediction_idx = np.argmax(new_grapheme_prediction, axis=0)
    vowel_prediction_idx = np.argmax(new_vowel_prediction, axis=0)
    consonant_prediction_idx = np.argmax(new_consonant_prediction, axis=0)

    grapheme_res = []
    for idx in range(len(grapheme_prediction_idx)):
        grapheme_res.append(new_grapheme_prediction[grapheme_prediction_idx[idx], idx])

    vowel_res = []
    for idx in range(len(vowel_prediction_idx)):
        vowel_res.append(new_vowel_prediction[vowel_prediction_idx[idx], idx])

    consonant_res = []
    for idx in range(len(consonant_prediction_idx)):
        consonant_res.append(new_consonant_prediction[consonant_prediction_idx[idx], idx])

    grapheme_prediction = np.reshape(grapheme_res, [n_instances, -1])
    vowel_prediction = np.reshape(vowel_res, [n_instances, -1])
    consonant_prediction = np.reshape(consonant_res, [n_instances, -1])

    # to categorical
    grapheme_prediction = np.argmax(grapheme_prediction, axis=1)
    vowel_prediction = np.argmax(vowel_prediction, axis=1)
    consonant_prediction = np.argmax(consonant_prediction, axis=1)

    # to single list
    for idx in tqdm(range(len(grapheme_prediction))):
        final_predictions.append(consonant_prediction[idx])
        final_predictions.append(grapheme_prediction[idx])
        final_predictions.append(vowel_prediction[idx])


# to csv
submission = pd.read_csv('../input/bengaliai-cv19/sample_submission.csv')
submission.target = np.hstack(final_predictions)
submission.to_csv('submission.csv', index=False)

submission.head(20)
