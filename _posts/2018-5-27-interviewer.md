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
  * Naive bayes classifier dựa trên công thức bayes: $ P(a/b) = \frac{P(a\cap b)P(a)}{P(b}$. Giả sử có 2 class (c1,c2), thì hiểu một cách đơn giản
  naive bayes là ta tìm xác suất của $P(c1/b)$,$P(c2/b)$ rồi so sánh 2 xác suất này xem cái nào lớn hơn thì sẽ thuộc về class đó. Vì P(b) là như
  nhau nên ta chỉ cần tính P(a\cap b) và P(a)
    * P(a\cap b) là xác suất của a,b cũng xảy ra
    * P(a) là xác xuất 
  
