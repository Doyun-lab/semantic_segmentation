# -*- coding: utf-8 -*-
"""Capstone_UNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YtXwOFR9KC0SSKezplvHQwHUOFzxiTZ3
"""

# Commented out IPython magic to ensure Python compatibility.
# # 필요한 패키지 다운로드
# %%capture
# !pip install tensorflow_addons
# !pip install albumentations
# !pip install segmentation_models
# !pip install keras
# !sudo apt install zip unzip
# 
# %env SM_FRAMEWORK=tf.keras
# 
# # 패키지 불러오기
# import numpy as np 
# import matplotlib.pyplot as plt
# 
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# 
# import tensorflow as tf
# import keras
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, UpSampling2D, Concatenate
# from keras.layers.normalization import BatchNormalization
# from keras.models import Sequential, Model, model_from_json, load_model
# from keras.layers.convolutional import Conv2D
# from keras.regularizers import l2
# from sklearn.model_selection import train_test_split
# 
# from segmentation_models.metrics import iou_score
# 
# from PIL import Image
# from scipy import ndimage
# from tqdm.notebook import tqdm
# 
# import cv2
# import multiprocessing
# import os

from google.colab import drive
drive.mount('/content/gdrive/')

# 평탄화 적용 경로
# orginal = "/content/gdrive/My Drive/dataset/Original_data_br"

# CLAHE Algorithm 적용 경로
# orginal = "/content/gdrive/My Drive/dataset/Original_data_br2"

# FLIP 적용 경로
# orginal = "/content/gdrive/My Drive/dataset/Original_data_flip"

# 폴더 경로 지정
orginal = "/content/gdrive/My Drive/dataset/Original_data"
label = "/content/gdrive/My Drive/dataset/Labeled_data"

# 경로에 있는 파일 이름 불러와서 저장
orginal_onlyfiles = [f for f in os.listdir(orginal) if os.path.isfile(os.path.join(orginal, f))]
label_onlyfiles = [f for f in os.listdir(label) if os.path.isfile(os.path.join(label, f))]

# 개수 확인
print("{0} original images".format(len(orginal_onlyfiles)))
print("{0} label images".format(len(label_onlyfiles)))

# 파일 이름 순서로 정렬
orginal_onlyfiles.sort()
label_onlyfiles.sort()

# 이미지 불러와서 thumbnail 함수로 비율 유지 변환 후 저장
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
    img = load_img(orginal + "/" + _file)  
    img.thumbnail((256, 512))
    x = img_to_array(img) 
    origin_img_arr.append(x)
for _file in label_files:
    img = load_img(label + "/" + _file)  
    img.thumbnail((256, 512))
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

# compile에 사용할 mIoU 정의
miou = MeanIoU(num_classes=32)

# U-net Architecture 정의

# conv_factory 정의 : 배치 정규화 + ReLU + Conv2D + 드랍아웃(옵션)
def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (5, 5), dilation_rate=(2, 2),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
  
    return x


# dense block 정의 : conv_factory의 nb_layers stack이 함께 병합
def denseblock(x, concat_axis, nb_layers, growth_rate, dropout_rate=None, weight_decay=1E-4):
    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,dropout_rate, weight_decay)
        list_feat.append(x)
    x = Concatenate(axis=concat_axis)(list_feat)

    return x


# dense block으로 수정된 U-Net 정의
def u_net():
    dr = 0.5
    nr = 2
    mod_inputs = Input((192,256,3))
    print("inputs shape:", mod_inputs.shape) 

    conv1 = Conv2D(64/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mod_inputs)
    print("conv1 shape:", conv1.shape)
    db1 = denseblock(x=conv1, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=dr)
    print("db1 shape:", db1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(db1)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(128/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    db2 = denseblock(x=conv2, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=dr)
    print("db2 shape:", db2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(db2)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(256/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    db3 = denseblock(x=conv3, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=dr)
    print("db3 shape:", db3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(db3)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(512/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    print("conv4 shape:", conv4.shape)
    db4 = denseblock(x=conv4, concat_axis=3, nb_layers=4, growth_rate=16, dropout_rate=dr)
    print("db4 shape:", db4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(db4)
    print("pool4 shape:", pool4.shape)

    conv5 = Conv2D(1024/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    print("conv5 shape:", conv5.shape)
    db5 = denseblock(x=conv5, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=dr)
    print("db5 shape:", db5.shape)
    up5 = Conv2D(512/nr, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db5))
    print("up5 shape:", up5.shape)
    merge5 = Concatenate(axis=3)([ BatchNormalization()(db4), BatchNormalization()( up5)]) 
    print("merge5 shape:", merge5.shape)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    print("conv6 shape:", conv6.shape)
    db6 = denseblock(x=conv6, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=dr)
    print("db5 shape:", db6.shape)
    up6 = Conv2D(256/nr, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(db6))
    print("up6 shape:", up6.shape)
    merge6 = Concatenate(axis=3)([BatchNormalization()(db3), BatchNormalization()(up6)]) 
    print("merge6 shape:", merge6.shape)

    conv7 = Conv2D(256/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    print("conv7 shape:", conv7.shape)
    db7 = denseblock(x=conv7, concat_axis=3, nb_layers=2, growth_rate=16, dropout_rate=dr)
    print("db7 shape:", db7.shape)
    up7 = Conv2D(128/nr, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(db7))
    print("up7 shape:", up7.shape)
    merge7 = Concatenate(axis=3)([BatchNormalization()(db2), BatchNormalization()(up7)]) 
    print("merge7 shape:", merge7.shape)

    conv8 = Conv2D(128/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    print("conv8 shape:", conv8.shape)
    db8 = denseblock(x=conv8, concat_axis=3, nb_layers=2, growth_rate=16, dropout_rate=dr)
    print("db8 shape:", db8.shape)
    up8 = Conv2D(64/nr, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(db8))
    print("up8 shape:", up8.shape)
    merge8 = Concatenate(axis=3)([BatchNormalization()(db1), BatchNormalization()(up8)]) 
    print("merge8 shape:", merge8.shape)

    conv9 = Conv2D(64/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    print("conv9 shape:", conv9.shape)
    db9 = denseblock(x=conv9, concat_axis=3, nb_layers=2, growth_rate=16, dropout_rate=dr)
    print("db9 shape:", db9.shape)
    conv10 = Conv2D(32/nr, 3, activation='relu', padding='same', kernel_initializer='he_normal')(db9) 
    print("conv10 shape:", conv10.shape)
    conv11 = Conv2D(3, 1, activation='sigmoid')(conv10)  
    print("conv11 shape:", conv11.shape)

    model = Model(inputs=mod_inputs, outputs=conv11) 

    # Model Compile
    model.compile(optimizer='adam', loss = 'MSE', metrics=[tf.keras.metrics.MeanIoU(num_classes=32), iou_score, miou.mean_iou])
    
    return model

# U-Net 불러오기
model = u_net()

# Model fitting
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
    plt.savefig('/content/gdrive/MyDrive/Unet_original/' + '%03d' % int(i) + '_pred.png')

    plt.show()

# Model Evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print(score)

