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
diện giương mặt đơn giản mà ta có thể thử. Bài này ta sẽ tìm hiểu về eigenface và cùng một model đơn giản với Opencv. Eigenface lấy ý tưởng đằng sau từ PCA, chắc cũng đã có nhiều người biết đến phương pháp này. PCA là một phương pháp giảm chiều dữ liệu, khi mà dữ liệu có chiều lớn mà chúng ta chỉ có thể visualize ở chiều nhỏ hơn 3 thì PCA sẽ là một phương pháp giúp ta đữa data về một không gian mới(ta gọi là PCA space) mà vẫn cố giữ lại được thông tin nhiều nhất có thể trên data. 
## 1, Tìm hiểu về PCA
## 2, Tìm hiểu EigenFace
## 3, Build model

## 1, Tìm hiểu về PCA
* PCA là một trong những phương pháp giảm chiều dữ liệu ( Dimensionality reduction techniques ) phổ biến nhất và được sử dụng trong nhiều lĩnh vực khác nhau. PCA có nhiều ứng dụng như tìm mối tương quan giữa các biến ( relationship between observation), trích xuất những thông
tin quan trọng từ data,phát hiện và loại bỏ outlier và giảm chiều chiều dữ liệu.Ý tưởng của phương pháp PCA là tìm ra một không gian mới
để chiếu(project) data sao cho variation giữ lại là nhiều nhất. Ta có thể hình dung qua hình vẽ dưới đây.

![pca1](/assets/images/pca1.jpg)
* Có 2 phương pháp tiếp cận PCA là covarian matrix và SVD chúng ta cùng tìm hiểu qua 2 phương pháp này .
* Phương pháp Covarian matrix : Các bước thực hiện thuật toán như sau :
 ![pca](/assets/images/pca.jpg)
 
  * X data có chiều MxN ( với N là số sample ,M là số feature).
    2, Tính mean của X :
  $$
  \mu = \frac{1}{N}\cdot\sum_{i=1}^{N}x_{i}
  $$
  * Trừ X với mean của X :
  $$
  D = \{d_{1},d_{2},..,d_{N}\} = \sum_{i=1}^{N}x_{i} - \mu
  $$
  * Tính toán covarian :
    $$
    \sum = \frac{1}{N-1}\cdot D\cdot D^{T}
    $$
  * Tính toán EigenVector `V` và EigenValue $\lambda$ của Covarian $\sum$
  * Sort EigenValue tương ứng với EigenVector theo thứ tự $\lambda$ giảm dần .
  * Chọn những EigenVector tương ứng với EigenValue lớn nhất $ W = \{v_{1},v_{2},..v_{k}\} $ . EigenVector W sẽ làm đại diện để project X vào PCA space
  * Tất cả sample X sẽ được project vào không gian nhỏ hơn theo công thưc $Y = W^{T}\cdot D$
* Lưu ý về dimension cái biến :

![dimension](/assets/images/dimension.jpg)



