---
layout: post
title: "Tìm hiểu eigenFace trong face recognite "
description: "nhận diện giương mặt "
categories: [python,machine_learning]
tags: [python,machine learning]
redirect_from:
  - /2018/04/19/
---
# Mở đầu .
Có bao giờ bạn vào facebook rồi một ngày nọ có một thông báo hiện lên bạn được tag trong một bước ảnh nào đó. Đã bao giờ bạn nghĩ làm sao 
facebook nhận diện ra mặt bạn, mình cũng không biết nữa vì tất cả thuật toán của nó là điều bí mật. Tuy vậy vẫn có nhiều phương pháp nhận 
diện giương mặt đơn giản mà ta có thể thử. Bài này ta sẽ tìm hiểu về eigenface và cùng một model đơn giản với Opencv. Eigenface lấy ý tưởng 
đằng sau từ PCA, chắc cũng đã có nhiều người biết đến phương pháp này. PCA là một phương pháp giảm chiều dữ liệu, khi mà dữ liệu có chiều lớn
mà chúng ta chỉ có thể visualize ở chiều nhỏ hơn 3 thì PCA sẽ là một phương pháp giúp ta đữa data về một không gian mới(ta gọi là PCA space)
mà vẫn cố giữ lại được thông tin nhiều nhất có thể trên data. Mỗi tấm ảnh mà
