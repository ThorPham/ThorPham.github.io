---
layout: post
published: true

title: "Drowsiness detection với Dlib và OpenCV"
description: "Drowness detection với Dlib và OpenCV."
categories: [computer_vision,python]
tags: [pyhon,computer_vision]
---
## Mở Đầu .
* Bài trước chúng ta đã tìm hiểu về facial landmark. Trong bài này chúng ta sẽ ứng dụng facial landmark vào Drowsiness detection. Drowness detection
dùng để xác định trạng thái ngủ gật hay không dựa vào facial landmark của eye. Thường được cái tài xế xử dụng khi điều khiển phương tiện giao
thông để hạn chế tai nạn.
* Cấu trúc của bài :
  * Tìm hiểu ý tưởng .
  * Xây dựng model .
  * Test model
## Tìm hiểu ý tưởng .
* Ý tưởng cũng rất đơn giản thôi, là chúng ta sẽ dựa vào facial landmark của eyes để xác định được tỉ lệ nào đó như một ngưỡng để xem xét
mắt đang nhắm hay mở.Trong paper **Real-Time Eye Blink Detection using Facial Landmarks** của **Tereza Soukupova** và **Jan ´ Cech** đã
tìm ra được một công thức giải quyết vấn đề này có tên gọi là eye aspect ratio(EAR).Chúng ta cùng tìm hiểu qua về công thức này.

$$
EAR =  \frac{||p_{2} - p_{6}|| + ||p_{3} - p_{5}||}{||p_{1} - p_{4}||}
$$

   * Trong đó $p_{i}$ là lankmark point của eye, ký hiệu **|| ||** là khoảng cách euclide.
![drowsiness1](/assets/images/drowness1.jpg)
* Đồ thị EAR,trong đó p1,p2,p3,p4,p5,p6 là landmark point của eye(lưu ý ta sẽ ký hiệu bắt đầu bằng 0 thay vì bằng 1 trong model).Biểu đồ bên dưới là đồ thị của EAR . Khi mà eye ta thấy là EAR sẽ nằm dưới threshold 0.15 và bình thường của nó sẽ lớn hơn 0.25. 
* Đó là ý tưởng của bài toán.Ở đây có 1 số lưu ý là :
   * Có 2 eye nên ta sẽ lấy trung bình của 2 eye để lấy EAR
   * Để tránh trường hợp nháy mắt hay hay detection sai ta sẽ cho EAR một khoảng thời gian đủ lâu để xác nhận là drowsiness.
   * Threshold sẽ do ta chọn theo ý muốn ta ta thấy hợp lý.
