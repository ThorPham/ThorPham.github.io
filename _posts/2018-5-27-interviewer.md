---
layout: post
title: "Một số câu hỏi interviewer AI,Deep learning,machine learning"
description: "Một số câu hỏi interviewer AI,Deep learning,machine learning"
categories: [computer_vision]
tags: [computer_vision,python]
redirect_from:
  - /2018/04/22/
---
* 1, Why is naive bayes called “naive” ?.
  * Tại vì trong naive bayes ta đã giải định các feature không có mối quan hệ với nhau,tức là xác suất xảy ra của feature này không bị ảnh
  hưởng bởi feature kia. Lấy vd : text = " bóng đá được xem là môn thể thao vua" thì xác xuất cụm từ "thể thao" xuất hiện không bị ảnh hưởng bởi
  cụm từ "bóng đá". Trên thực tế điều này không đúng ha vì 2 cụm từ này có mối quan hệ tương đối mật thiết với nhau. Đây cũng là một nhược điểm
  của thuật toán này.
* 2, Tell me about naive bayes classifier ?
  * Naive bayes classifier dựa trên công thức bayes: $ P(a/b) = \frac{P(a\cap b)P(a)}{P(b)}$. Giả sử có 2 class (c1,c2), thì hiểu một cách đơn giản naive bayes là ta tìm xác suất của $P(c1/b)$,$P(c2/b)$ rồi so sánh 2 xác suất này xem cái nào lớn hơn thì sẽ thuộc về class đó. Vì P(b) là như nhau nên ta chỉ cần tính #P(a\cap b)# và $P(a)$
* 3, Explain TF-IDF ?
* 4, What are word2vec vectors?
* 5, How does SVM learns non-linear boundaries ? Explain.
* 6, What is precision and recall ? Which one of this do you think is important in medical diagnosis ?
* 7, Define precision and recall ?
* 8, What is random about Random Forest ?
* 9, What are the criteria for splitting at a node in decision trees ?
* 10, What is the advantage with random forest ?
* 11, Tell me about boosting algorithms ?
* 12, How does gradient boosting works ?
* 13, What are the kernels used in SVM ? What is the optimization technique of SVM ?
* 14, How do you decide K in K-Means clustering algorithm ?
* 15, Can you tell DB-SCAN algorithm ?
* 16, How does HAC (Hierarchical Agglomerative clustering) work ?
* 17, Explain PCA ? Tell me the mathematical steps to implement PCA ?
* 18, What is disadvantage of using PCA ?
* 19, How does CNN work ? Explain the implementation details ?
* 20, What is the range of sigmoid function ?
* 21, What is mean and variance of standard normal distribution ?
* 22, Which model would you use in case of unbalanced dataset: Random Forest or Boosting ? Why ?
* 23, What are Lasso and Ridge regression ?
* 24, What is Gaussian Mixture model ? How does it perform clustering ?
* 25, How is Expectation Maximization performed ? Explain both the steps ?
* 26, Explain the intuition behind BIC or AIC ?
* 27, What’s the trade-off between bias and variance?
* 28, What is the difference between supervised and unsupervised machine learning?
* 29,  How is KNN different from k-means clustering?
* 30, Explain how a ROC curve works.
* 31, Explain the difference between L1 and L2 regularization.
* 32, What’s the difference between Type I and Type II error?
* 33, What’s the difference between probability and likelihood?
* 34, What’s the difference between a generative and discriminative model?
* 35, How is a decision tree pruned?
* 36, Which is more important to you– model accuracy, or model performance?
* 37, What’s the F1 score? How would you use it?
* 37,Explain max un-pooling operation for increasing the resolution of feature maps.
* 38,What is a Learnable up-sampling or Transpose convolution ?
* 39,Describe the transition between R-CNN, Fast R-CNN and Faster RCNN for object detection.
* 40,Describe how RPNs are trained for prediction of region proposals in Faster R-CNN?
* 41,Describe the approach in SSD and YOLO for object detection. How these approaches differ from Faster-RCNN. When will you use one over the other?
* 42,Difference between Inception v3 and v4. How does Inception Resnet compare with V4.
* 43,Explain main ideas behind ResNet? Why would you try ResNet over other architectures?
* 44, Explain batch gradient descent, stochastic gradient descent and mini-match gradient descent.
* 45, Explain Dropout and Batch Normalization. Why BN helps in faster convergence?
* 46 , 
