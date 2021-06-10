# 필요한 패키지 다운로드
%%capture
!pip install tensorflow_addons
!pip install albumentations
!pip install segmentation_models
!pip install keras
!sudo apt install zip unzip

%env SM_FRAMEWORK=tf.keras

# 패키지 불러오기
import os
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import numpy as np 
import pandas as pd
import random

from sklearn.model_selection import train_test_split

from PIL import Image
from functools import partial

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as keras

from glob import glob

import albumentations as A
import tensorflow_addons as tfa

from segmentation_models import Unet
from segmentation_models.metrics import iou_score

from scipy import ndimage
from tqdm.notebook import tqdm

import cv2
import cv2 as cv2
import multiprocessing
import os


# 폴더 경로 지정
orginal = "/content/gdrive/My Drive/Original_data"
label = "/content/gdrive/My Drive/Labeled_data"

# 경로에 있는 파일 이름 불러와서 저장
orginal_onlyfiles = [f for f in os.listdir(orginal) if os.path.isfile(os.path.join(orginal, f))]
label_onlyfiles = [f for f in os.listdir(label) if os.path.isfile(os.path.join(label, f))]

# 개수 확인
print("{0} original images".format(len(orginal_onlyfiles)))
print("{0} label images".format(len(label_onlyfiles)))

# 파일 이름 순서로 정렬
orginal_onlyfiles.sort()
label_onlyfiles.sort()

# 이미지 불러와서 Traget Size로 변환 후 저장
origin_files = []
label_files = []
origin_img_arr = []
label_img_arr = []

i=0
for _file in orginal_onlyfiles:
    origin_files.append(_file)
for _file in label_onlyfiles:
    label_files.append(_file)

for _file in origin_files:
    img = load_img(orginal + "/" + _file,target_size=(224,224)) 
    x = img_to_array(img) 
    origin_img_arr.append(x)
for _file in label_files:
    img = load_img(label + "/" + _file,target_size=(224,224)) 
    x = img_to_array(img) 
    label_img_arr.append(x)

# array로 저장
origin_arr = np.array(origin_img_arr)
label_arr = np.array(label_img_arr)

# Train / Validation / Test 를 6 : 2 : 2 로 나누기
x_train, x_temp, y_train, y_temp = train_test_split(origin_arr, label_arr, test_size=0.4, random_state=234)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=0.5, random_state=234)

# 정규화
x_train = x_train/255
y_train = y_train/255
x_val = x_val/255
y_val = y_val/255
x_test = x_test/255
y_test = y_test/255


# MeanIoU 함수 정의
class MeanIoU(object):
    """Mean intersection over union (mIoU) metric.
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    The mean IoU is the mean of IoU between all classes.
    Keyword arguments:
        num_classes (int): number of classes in the classification problem.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

    def mean_iou(self, y_true, y_pred):
        """The metric function to be passed to the model.
        Args:
            y_true (tensor): True labels.
            y_pred (tensor): Predictions of the same shape as y_true.
        Returns:
            The mean intersection over union as a tensor.
        """

        return tf.compat.v1.py_func(self._mean_iou, [y_true, y_pred], tf.float32)

    def _mean_iou(self, y_true, y_pred):
        """Computes the mean intesection over union using numpy.
        Args:
            y_true (tensor): True labels.
            y_pred (tensor): Predictions of the same shape as y_true.
        Returns:
            The mean intersection over union (np.float32).
        """

        target = np.argmax(y_true, axis=-1).ravel()
        predicted = np.argmax(y_pred, axis=-1).ravel()

        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2
        )
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape(
            (self.num_classes, self.num_classes)
        )

        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 1

        return np.mean(iou).astype(np.float32)
    
miou = MeanIoU(num_classes=32)



##### FCN-32 ######

IMAGE_ORDERING = 'channels_last'
def FCN32( n_classes ,  input_height=224, input_width=224 , vgg_level=3):

    assert input_height%32 == 0
    assert input_width%32 == 0


    img_input = Input(shape=(input_height,input_width,3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
    f5 = x

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense( 1000 , activation='sigmoid', name='predictions')(x)

    vgg  = Model(  img_input , x  )


    o = f5

    o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)
    o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    o = Dropout(0.5)(o)

    o = ( Conv2D( n_classes ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
    o = Conv2DTranspose( n_classes , kernel_size=(32,32) ,  strides=(32,32) , use_bias=False ,  data_format=IMAGE_ORDERING )(o)

    o = (Activation('sigmoid'))(o)
    model = Model( img_input , o )


    return model


#모델 정의
model = FCN32( 3 ,  input_height=224, input_width=224 , vgg_level=3)


#model compile
model.compile(optimizer='adam', loss="MSE",  metrics=[miou.mean_iou])

# Model Fitting
hist = model.fit(x_train, y_train, epochs=100, shuffle = True, batch_size= 10, validation_data=(x_val, y_val))

# Model Predict
pred = model.predict(x_test[0:140,:,:,:])


# 시각화
for i in range(0,np.shape(pred)[0]):
    
    fig = plt.figure(figsize=(20,8))
    
    # 실제 사진 
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(x_test[i])
    ax1.title.set_text('Actual frame')
    ax1.grid(b=None)
    
    # 라벨 사진
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('Ground truth labels')
    ax2.imshow(y_test[i])
    ax2.grid(b=None)
    
    # 예측 사진
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('Predicted labels')
    ax3.imshow(pred[i])
    ax3.grid(b=None)
    
    # 원하는 경로에 그림 저장
    plt.savefig('/content/gdrive/MyDrive/segnet_vgg16/' + '%03d' % int(i) + '_pred.png')

    plt.show()

