---
layout: post
title: "Nhận dạng chữ số viết tay với sklearn và opencv"
description: "nhận dạng chữ số viết tay với HOG và SVM"
categories: [Machine_learning]
tags: [random, jekyll]
redirect_from:
  - /2018/04/6/
---
## Mở đầu
Có rất nhiều thuật toán để nhận diện chữ số viết tay (hand writen digit) như neural network,CNN với độ chính xác rất cao lên tới 99%. Nhưng cũng không nên phủ nhận các thuật toán áp dụng theo kiểu truyền thống .Trong bài viết này chúng ta cùng tìm hiểu 
Hog(histogram of oriented gradient) và svm( support vector machine) để nhận dạng chữ số viết tay trên bộ dữ liệu MNIST.
* Cấu trúc bài viết:
  * Xây dựng model nhận diện digit.
  * Predict trên ảnh có nhiều ditgit
## Xây dựng model nhận diện digit.
* Dữ liệu mà chúng ta training model là mnist. Đây có thể coi là "hello word" trong machine learning. Trước hết ta tìm hiểu sơ qua về dữ liệu, mnist là tập hợp các ảnh xám về digit có chiều là 28x28 bao gồm 70.000 ngàn ảnh, trong đó có 60.000 ảnh để training và 10.000 ảnh để testing. Ý tưởng của model là dùng HOG(histogram oriented of gradient) để extract feature( các bạn có thể coi các bài trước để biết thêm hog là gì). Sau khi có feature ta sẽ đưa vào model SVM để phân loại. Cuối cùng dùng opencv đế segmentation digit và dùng model chúng ta vừa build để predict. Bắt tay vào model nào.
