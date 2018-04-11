---
layout: post
title: "Nhận diện pedestrian với window search"
description: "kết hợp hog ,svm với window search"
categories: [demo]
tags: [demo, jekyll]
redirect_from:
  - /2018/04/11/
---
~~~ ruby
import numpy as np
import cv2
from skimage import io
from skimage.feature import hog
import glob
from matplotlib import pyplot as plt
%matplotlib inline
~~~
Tạo một function để tính hog

~~~ ruby
def hog_feature(image):
    feature_hog = hog(image,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm="L2")
    return feature_hog
~~~

Preprocessing data
~~~ ruby
path_pos = glob.glob("./pedestrians128x64/"+"*.ppm")
X_pos = []
y_pos = []
for path in path_pos :
    im = io.imread(path,as_grey=True)
    im_feature = hog_feature(im)
    X_pos.append(im_feature)
    y_pos.append(1)
~~~

Tạo negative sample
~~~ ruby
path_neg = glob.glob("./pedestrians_neg/"+"*.jpg")
X_neg = []
y_neg = []
w = 64
h = 128
for path in path_neg :
    im = io.imread(path,as_grey=True)
    for j in range(15):
        x = np.random.randint(0,im.shape[1]-w)
        y = np.random.randint(0,im.shape[0]-h)
        im_crop = im[y:y+h,x:x+w]
        im_feature = hog_feature(im_crop)
        X_neg.append(im_feature)
        y_neg.append(-1)
~~~
stack pos và neg 
~~~ ruby
X_pos = np.array(X_pos)
X_neg = np.array(X_neg)
X_train = np.concatenate((X_pos,X_neg))
y_pos = np.array(y_pos)
y_neg = np.array(y_neg)
y_train = np.concatenate((y_pos,y_neg))
~~~
Training model
~~~ ruby
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC()
model.fit(X_train,y_train)
y_predict = model.predict(X_train)
print(accuracy_score(y_train,y_predict))
~~~
 Detection trên ảnh lớn
 Tạo 1 sliding window
 ~~~ ruby
 def sliding_window(image,window_size,step_size):
    for y in range(0,image.shape[0]-window_size[1],step_size[1]):
        for x in range(0,image.shape[1]-window_size[0],step_size[0]):
            roi = image[y:y+window_size[1],x:x+window_size[0]]
            yield (x,y,roi)
 ~~~
 ~~~ruby
 def overlaping_area(detection_1,detection_2):
    #detection_1,detection_2 format [x_left_top,y_left_top,score,width,height]
    x_1 = detection_1[0]
    y_1 = detection_1[1]
    x_w_1 = detection_1[0] + detection_1[3]
    y_h_1 = detection_1[1] + detection_1[4]
    
    x_2 = detection_2[0]
    y_2 = detection_2[1]
    x_w_2 = detection_2[0] + detection_2[3]
    y_h_2 = detection_2[1] + detection_2[4]
    # tính overlap theo ox,oy .Nếu ko giao nhau trả về 0
    overlap_x = max(0,min(x_w_1,x_w_2) - max(x_1,x_2))
    overlap_y = max(0,min(y_h_1,y_h_2) - max(y_1,y_2))
    # tính area overlap
    overlap_area = overlap_x*overlap_y
    # tính total area hợp của 2 detection
    total_area = detection_1[3]*detection_1[4] + detection_2[3]*detection_2[4] - overlap_area
    
    return overlap_area/float(total_area)
 ~~~
 
 ~~~ruby
 def nms(detections,threshold =0.5):
    # decections format [x_left_top,y_left_top,score,width,height]
    # nếu area overlap lớn hơn threshold thì sẽ remove detection nào có score nhỏ hơn
    if len(detections)==0:
        return []
    # sort detection theo score
    detections = sorted(detections,key = lambda detections : detections[2],reverse = True)
    #create new detection
    new_detections = []
    new_detections.append(detections[0])
    del detections[0]
    for index,detection in enumerate(detections):
        for new_detection in new_detections:
            if overlaping_area(detection,new_detection)> threshold : #compare areaoverlap với threshold
                del detections[index]
                break
        else :
            new_detections.append(detection)
            del detections[index]
    return new_detections
~~~

