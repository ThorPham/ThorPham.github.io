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
