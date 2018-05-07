---
layout: post

title: "Project multi digit recognition"
description: "Project multi digit recognition"
categories: [Machine_learning]
tags: [opencv,python]
---
* Model Multi digit Recognition . (HOG + SVM accuracy có 90%) trong khi đó CNN đạt 99%. Model này mình đã thêm "image data generation" 
để tránh trường hợp digit bị skew. Tuy vậy khi đưa vào multi recognite thì số 1 toàn bị fail .Chưa tìm ra được nguyên nhân để khắc phục. 
Có lẽ là do phần image processing trước khi nhận dạng. Đã cố gắng padding cho nó 1 khoảng rồi mà vẫn không được.
* Video [demo](https://www.youtube.com/watch?v=yO2IhxgKKLI)
* Code on gihub : [code](https://github.com/ThorPham/Multi-digit-recognite)
