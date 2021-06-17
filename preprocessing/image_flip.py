# 패키지 불러오기
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
import random
import re
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys

import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from scipy import ndimage
from tqdm.notebook import tqdm

import cv2
import multiprocessing
import os

from google.colab import drive
drive.mount('/content/gdrive/')

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
    img = load_img(orginal + "/" + _file,target_size=(384,384)) 
    x = img_to_array(img) 
    origin_img_arr.append(x)
for _file in label_files:
    img = load_img(label + "/" + _file,target_size=(384,384)) 
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

# Train set에 대해서만 Flip 진행
x_train_flip = []
for imgx_t in x_train:
  img2 = array_to_img(imgx_t)
  img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
  img_arr = img_to_array(img2)
  img_arr = img_arr/255
  x_train_flip.append(img_arr)

y_train_flip = []
for imgy_t in y_train:
  img2 = array_to_img(imgy_t)
  img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
  img_arr = img_to_array(img2)
  img_arr = img_arr/255
  y_train_flip.append(img_arr)

# 기존 Train set과 합치기
x_train_f = np.array(x_train_flip)
y_train_f = np.array(y_train_flip)

x_train_m = np.concatenate([x_train, x_train_f])
y_train_m = np.concatenate([y_train, y_train_f])