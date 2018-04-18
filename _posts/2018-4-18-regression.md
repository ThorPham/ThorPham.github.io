---
layout: post
title: "Tìm hiểu regression trong object detection "
description: "phân tích cảm xúc"
categories: [python,machine_learning]
tags: [python,machine learning]
redirect_from:
  - /2018/04/18/
---
* ở đầu 
Lần đầu tiên mình đọc về thuật toán YOLO(you look only one) là trên khóa "Convolution neural network" của thầy Andrew Ng trên coursera.
Có hàng ngàn câu hỏi vì sao ở trong đầu mình hiện ra dù đi hỏi khắp nơi mà nhiều trong số đó vẫn chưa có lời giải đáp thỏa mãn mình. Trong
đó có key word `Bounding-box regression`, mình suy nghĩ rất nhiều, đọc cũng kha khá bài viết trên mạng mà vẫn không hiểu nổi. Một câu hỏi cứ
lởn vởn trong đầu mình là các `bouding box` trong thuật toán yolo được tạo ra như thế nào ta, trước giờ mình chỉ dùng regression để predict 
các biến liên tục vậy họ áp dụng để detection bounding box ra sao. Người ta build yolo là tổng hợp của rất nhiều thuật toán tạo nên bộ xương
cho yolo .Thiết nghĩ những người mới lần đầu tập tọe vào deep learning như mình thì nên chia yolo từng phần để xử lý có lẽ sẽ dễ thở hơn. Trong
bài hôm nay mình sẽ làm rõ `bounding box` được tạo ra từ regression như thế nào bằng một ví dụ rất đơn giản.
* Các bươc thực hiện :
  * Chuẩn bị dữ liệu .
  * Traing model .
  * Đánh giá model
## Chuẩn bị dữ liệu .
Dữ liệu 'input` là những image có object mà ta muốn detection và `ouput` là những bouding box sẽ có dạng (x,y,w,h). Trong đó x,y là tọa độ
leftop của bounding box, (w,h) là width và height. Chúng ta sẽ mô phỏng dữ liệu như sau :
~~~ ruby
np.random.seed(10)
number_data = 5000
img_size = 8
min_size_obj = 1
max_size_obj = 4
number_obj = 2
# x là dataset image, y là label với 4 tham số(x,y,w.h)
bboxes = np.zeros((5000,2,4))
image = np.zeros((5000,img_size,img_size))
for i in range(5000):
    for obj in range(number_obj):
        w,h = np.random.randint(min_size_obj,max_size_obj,size = 2)
        x = np.random.randint(0,img_size-w)
        y = np.random.randint(0,img_size-h)
        bboxes[i,obj,:] = (x,y,w,h)
        image[i,y:y+h,x:x+w] = 1
~~~
* Giải thích một tí :
  * Ta sẽ tạo 5000 image có size (8,8) `image = np.zeros((5000,img_size,img_size))`. Image sẽ có background là white
  * 5000 `bounding box` có size từ w,h từ 1-4 và có màu đen
  * Mỗi image chỉ có duy nhất 1 object
* Image sau khi tạo sẽ như thế này :
![bounding_box](/assets/images/bounding.jpg)
* Cái chúng ta cần predict là đường viền màu đỏ. Image sẽ có dimension là (5000, 8, 8) ,bounding box có dimension là (5000, 1, 4).
