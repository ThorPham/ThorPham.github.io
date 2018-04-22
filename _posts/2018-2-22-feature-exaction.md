---
layout: post
title: "Feature extraction trong computer vision"
description: "tìm hiểu về feature extraction trong computer vision"
categories: [computer_vision]
tags: [computer_vision,python]
redirect_from:
  - /2018/04/22/
---
## Mở đầu
## Local binary pattern
* Local binary pattern nó là một thuật toán mô tả texture(cầu trúc) của một image. Ý tưởng cơ bản của nó là mô phỏng lại cấu trúc cục bộ
(local texture) của image bằng cách so sánh mỗi pixel với các pixel lân cận nó(neighbors).Ta sẽ đặt một pixel là trung tâm(center) và so sánh
với các pixel lân cận với nó, nếu pixel trung tâm lớn hơn hoặc bằng pixel lân cận thì nó sẽ trả về giá trị 1, ngược lại 0. Ví dụ chúng ta
lấy bán kính 8 pixel lân cận thì lbp sẽ có dạng 11001111, là một chuỗi nhị phân để đơn giản và dễ đọc hơn ta sẽ chuyển về dạng decimal 207.

![LBP](/assets/images/lbp.jpg)

* Cách tính này có hạn chế đó là chỉ giới hạn 3x3 pixel không đủ để mô tả các cấu trúc large scale nên người ta mở rộng khái niệm LBP bằng cách định nghĩa thêm 2 tham số là (P,R) trong đó P là số pixel lân cận xem xét và R là bán kính ta quét từ pixel trung tâm. Như hình bên dưới.
![LBP2](/assets/images/lbp2.jpg)
* Công thức LBP như sau :

$$
LBP_{r,p} = \sum_{n=0}^{p-1}S(X_{r,p,n}-X_{p})2^{n}
$$

 trong đó :
 
 $$ 
 S(x) =  \begin{cases}
  1, & \text{if } x >= 1, \\
  0, & \text{otherwise}.
\end{cases}
 $$
