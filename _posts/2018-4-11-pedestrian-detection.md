---
layout: post
title: "Nhận diện pedestrian với window search"
description: "kết hợp hog ,svm với window search"
categories: [demo]
tags: [python,machine learning]
redirect_from:
  - /2018/04/11/
---
Object regconite bao gồm 2 phần việc đó là object classifier và  object detecter. Hiểu một cách đơn giản đó là nếu chúng ta muốn máy tính nhận dạng được con mèo hay con chó thì trước tiên nó sẽ phải detecter đối tượng đó trên image và sau đó xem đối tượng đó là cái gì bằng cách classifier .Với sự phát triển của deep learning như hiện nay đã có rất nhiều thuật toán giúp ta giải quyết vấn đề này như R-CNN,Fast or Faster R-CNN,YOLO hay SSD với tốc độ xử lý nhanh và độ chính xác cao. Tuy vậy những cách truyền thống vẫn là sự lựa chon tốt khi mà chúng ta có ít dữ liệu và muốn build một model nào đó đơn giản hơn những cái phức tạp hơn như deep learning. Trong bài này chúng ta sẽ nhận diện pedestrian bằng phương pháp cổ điển trong computer vision và sau đó bạn có thể tự build một model custom nào đó theo ý của bạn .Thuật toán sử dụng trong bài là HOG + SVM + Window search

Cách bước thực hiện ta chia làm 2 giai đoạn tương ứng với classifier và detecter :
* Giai đoạn 1 classifier
1, Chuẩn bị dữ liệu
2,Trích chọn đặc trưng
3,Build model
4,Đánh giá và cải thiện model
* Gia đoạn 2  Detecter
1, Xây dựng sliding window
2, Xây dựng NMS(non-maxinum-suppression)
3, Detecter
# Giai đoạn 1 classifier
1, Chuẩn bị dữ liệu
Dữ liệu chúng ta cần chuẩn bị gồm 2 phần . Một là positive sample ( gọi tắt là pos) là data pedestrian và chúng ta gắn label cho nó là 1. Thứ hai là negative sample (Neg) là dữ liệu không chứa pedestrian bạn có thể lấy như background, car, house ... và ta gắn nhãn là 0.(lưu ý nếu training trong opecv thì nhãn gắn bắt buộc là 1 và -1 ).
~~~ ruby
# image positive
path_pos = glob.glob("./pedestrians128x64/"+"*.ppm")
plt.subplots(figsize =(10,5))
for i in range(6):
    image1 = io.imread(path_pos[i])
    plt.subplot(1,6,i +1)
    io.imshow(image1)
# image negative
path_neg = glob.glob("./pedestrians_neg/"+"*.jpg")
~~~

![pedestrian](/assets/images/pedestian1.jpg)
