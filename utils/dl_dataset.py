from sklearn.datasets import fetch_openml
from PIL import Image
import numpy as np
import ssl
import os

data_dir = "../data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

#セキュリティリスク有
ssl._create_default_https_context = ssl._create_unverified_context

mnist = fetch_openml('mnist_784', version=1, data_home="../data/")


data_dir_path = "../data/img_28/"
if not os.path.exists(data_dir_path):
    os.mkdir(data_dir_path)

# データの取り出し
X = mnist.data
y = mnist.target
max_num = 200
count7 = 0
count8 = 0
for i in range(len(X)):
    if (y[i] is "7") and (count7<max_num):
        file_path="../data/img_28/img_"+y[i]+"_"+str(count7)+".jpg"
        im_f=(X[i].reshape(28, 28))
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))
        pil_img_f.save(file_path)
        count7 += 1

    if (y[i] is "8") and (count8<max_num):
        file_path="../data/img_28/img_"+y[i]+"_"+str(count8)+".jpg"
        im_f=(X[i].reshape(28, 28))
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))
        pil_img_f.save(file_path)
        count8 += 1

    if (count7>=max_num) and (count8>=max_num):
        break

data_dir_path = "../data/img_28_test/"
if not os.path.exists(data_dir_path):
    os.mkdir(data_dir_path)

# 上記で7,8の画像を作成するのに使用したindexの最終値
i_start = i+1
print(i_start)

count2=0
count7=0
count8=0
max_num=5

for i in range(i_start,len(X)):

    if (y[i] is "2") and (count2<max_num):
        file_path="../data/img_28_test/img_2_"+str(count2)+".jpg"
        im_f=(X[i].reshape(28, 28))
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))
        pil_img_f.save(file_path)
        count2+=1

    if (y[i] is "7") and (count7<max_num):
        file_path="../data/img_28_test/img_7_"+str(count7)+".jpg"
        im_f=(X[i].reshape(28, 28))
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))
        pil_img_f.save(file_path)
        count7+=1

    if (y[i] is "8") and (count8<max_num):
        file_path="../data/img_28_test/img_8_"+str(count8)+".jpg"
        im_f=(X[i].reshape(28, 28))
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))
        pil_img_f.save(file_path)
        count8+=1