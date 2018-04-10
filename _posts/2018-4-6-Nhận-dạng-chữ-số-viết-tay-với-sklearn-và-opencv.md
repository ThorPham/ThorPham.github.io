---
layout: post
title: "Nhận dạng chữ số viết tay với sklearn và opencv"
description: "nhận dạng chữ số viết tay với HOG và SVM"
categories: [Machine_learning]
tags: [random, jekyll]
redirect_from:
  - /2018/04/6/
---
Có rất nhiều thuật toán để nhận diện chữ số viết tay (hand writen digit) như neural network,CNN.Trong bài viết này chúng ta cùng tìm hiểu 
Hog(histogram of oriented gradient) và svm( support vector machine) để nhận dạng chữ số viết tay trên bộ dữ liệu MNIST.Trước tiên ta cùng 
tìm hiểu về hog.
#HOG là gì ?
#Code python
~~~ ruby
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
~~~ 
Load daset
~~~ ruby
(X_train,y_train),(X_test,y_test) = mnist.load_data()

~~~


~~~ ruby
X_train_feature = []
for i in range(len(X_train)):
    feature = hog(X_train[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1))
    X_train_feature.append(feature)
X_train_feature = np.array(X_train_feature,dtype = np.float32)

X_test_feature = []
for i in range(len(X_test)):
    feature = hog(X_test[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1))
    X_test_feature.append(feature)
X_test_feature = np.array(X_test_feature,dtype=np.float32)
~~~
~~~ ruby
model = LinearSVC(C=10)
model.fit(X_train_feature,y_train)
y_pre = model.predict(X_test_feature)
print(accuracy_score(y_test,y_pre))
~~~

~~~ ruby
image = cv2.imread("test1.png")
im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
_,contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(cnt) for cnt in contours]
~~~
![digit](https://github.com/ThorPham/thorpham.github.io/blob/master/assets/images/screenshots/digit.jpg)
~~~ ruby
for i in contours:
    (x,y,w,h) = cv2.boundingRect(i)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    roi = thre[y:y+h,x:x+w]
    roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    nbr = model.predict(np.array([roi_hog_fd], np.float32))
    cv2.putText(image, str(int(nbr[0])), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    cv2.imshow("image",image)
cv2.waitKey()
cv2.destroyAllWindows()
~~~
![digit_nopad](https://github.com/ThorPham/thorpham.github.io/blob/master/assets/images/screenshots/image_no_pand.jpg)
![digit_pad](https://github.com/ThorPham/thorpham.github.io/blob/master/assets/images/screenshots/image_pand.jpg)

