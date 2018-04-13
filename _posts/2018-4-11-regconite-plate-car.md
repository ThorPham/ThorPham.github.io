---
layout: post
title: "Nhận dạng biển số xe với opencv"
description: "nhận dạng biển số xe bằng image processing"
categories: [uncategorized]
tags: [random, jekyll]
---
Nhận dạng biển số xe chắc không còn xa lạ đối với chúng ta, hàng ngày đi gửi xe ở các chung cư hay trung tâm thương mại chúng ta đều có thể nhìn thấy một anh bảo vệ ngồi gần một chiếc máy tính soi đi soi lại trên màn hình cái gì đó, đôi khi bảo chúng ta tắt đèn xe mà chúng ta chẳng hiểu để làm gì. Thực ra là có 1 camera ở phía sau chụp lại biển số xe của chúng ta. Anh ta đang xem lại ảnh trên máy tính có mờ hay nhiễu gì không để máy tính có thể nhận dạng được các con số trên biển số xe của chúng ta. Trong bài này chúng ta sẽ tìm hiểu cách mà máy tính có thể nhận dạng được các con số hay chữ cái. Có rất nhiều phương pháp và thuật toán có thể giải quyết được vấn đề này từ những thuật toán machine learning hay những thuật toán hiện đại hơn là CNN + RNN trong deep learning.
Các bước thực hiện :
1, Nhận diện được vị trí của biển số xe trên image ( Object Localization)
2, Segmentation các kí tự trên biển số xe
3, Nhận dạng
Hai bước khó nhất là bước 1 và bước 2. Có một điểm chúng ta cần lưu ý là ở đây là camera đã đặt cố định và các character trên biển số xe
là tách biệt với nhau nên ta có thể dùng image procssing thông thường để lấy vị trí. Khác với những trường hợp nhận dạng realtime thì kỹ
thuật này không đạt được hiểu quả cao vì sẽ có rất nhiều nhiễu và image có rất nhiều hình thái khác nhau( rotation,scale..) nên khuyến nghị dùng deep learning sẽ hiểu quả hơn.
Một số biển số xe lấy trên mạng .


![car](https://github.com/ThorPham/thorpham.github.io/blob/master/assets/images/image1.jpg)
