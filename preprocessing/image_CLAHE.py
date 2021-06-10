from __future__ import print_function
import os
import cv2
import numpy as np
from PIL import Image
import multiprocessing as mp # 병렬처리 패키지1
import parmap # 병렬처리 패키지2

# 이미지 명암대비 조절 함수 정의(CLAHE)
def img_Contrast(image_path):
  image = cv2.imread(image_path)
  # 이미지를 LAB Color 로 변환
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  # 각 채널 분리
  l, a, b = cv2.split(lab)
  # L 채널에 CLAHE 적용
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
  cl = clahe.apply(l)
  # CLAHE 적용한 L 채널을 나머지 채널과 병합
  limg = cv2.merge((cl, a, b))
  # -----Converting image from LAB Color model to RGB model--------------------
  # LAB Color에서 RGB로 변환
  final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
  return final

# 파일 경로 지정
image_dir = './dataset/Original_data/'

# 파일 내부 파일들로 리스트 생성
image_list = os.listdir(image_dir)

# 새로운 폴더 생성(이미 있다면 지나가기)
try:
  os.makedirs('./dataset/tf_data2')
except OSError:
  print("File exists: './dataset/tf_data2'")

# 저장하는 경로
tf_dir = './dataset/tf_data2/'

num_cores = mp.cpu_count() # cpu 코어 개수

# 파일 리스트를 사용하여 이미지 불러와 CLAHE 적용하는 함수 정의
def transform_img(image_list): #
    for i in image_list:
        image_path = image_dir + i
        tf_image = img_Contrast(image_path)
        cv2.imwrite(os.path.join(tf_dir , i), tf_image)

# 코어 개수 단위로 파일 리스트 분리
splited_data = np.array_split(image_list, num_cores)
splited_data = [x.tolist() for x in splited_data]

# 분리된 리스트에서 각각 병렬처리하여 CLAHE 적용
parmap.map(transform_img, splited_data, pm_pbar = True, pm_processes = num_cores)
