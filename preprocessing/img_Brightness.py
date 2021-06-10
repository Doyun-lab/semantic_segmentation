from __future__ import print_function
import os
import cv2
import numpy as np
from PIL import Image
import multiprocessing as mp # 병렬처리 패키지1
import parmap # 병렬처리 패키지2

# Image control 함수 정의
def image_control(alpha_num, beta_num, image_path): # 알파 값, 베타 값, 이미지 경로
  image = cv2.imread(image_path) # 경로로부터 사진 불러오기
  new_image = np.zeros(image.shape, image.dtype) # 조정한 이미지를 저장하기 위해 작성

  alpha = alpha_num # Simple contrast control(기울기)
  beta = beta_num    # Simple brightness control(단순 명암 제어)
  
  for y in range(image.shape[0]):
      for x in range(image.shape[1]):
          for c in range(image.shape[2]):
              new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255) # 명암비 조절

  # 히스토그램 스트레칭
  dst = cv2.normalize(new_image, None, 0, 255, cv2.NORM_MINMAX)

  return dst

# 파일 경로 지정
image_dir = './dataset/Original_data/'

# 파일 내부 파일들로 리스트 생성
image_list = os.listdir(image_dir)

# 새로운 폴더 생성(이미 있다면 지나가기)
try:
  os.makedirs('./dataset/tf_data')
except OSError:
  print("File exists: './dataset/tf_data'")

# 저장하는 경로
tf_dir = './dataset/tf_data/'

num_cores = mp.cpu_count() # cpu 코어 개수

# 파일 리스트를 사용하여 이미지 불러와 명암비 조정, 히스토그램 스트레칭 적용하는 함수 정의
def transform_img(image_list):
    for i in image_list:
        image_path = image_dir + i
        tf_image = image_control(1.5, 40, image_path)
        tf_image = Image.fromarray(tf_image)
        tf_image.save(tf_dir + i)

# 코어 개수 단위로 파일 리스트 분리
splited_data = np.array_split(image_list, num_cores)
splited_data = [x.tolist() for x in splited_data]

# 분리된 리스트에서 각각 병렬처리하여 명암비 조정, 히스토그램 스트레칭
parmap.map(transform_img, splited_data, pm_pbar = True, pm_processes = num_cores)
