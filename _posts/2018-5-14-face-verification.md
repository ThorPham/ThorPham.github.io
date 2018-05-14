---
layout: post
title: "Project face verification 12 singer việt nam với facenet"
description: "Project face verification 12 singer việt nam với facenet"
categories: [demo]
tags: [python,machine learning]
redirect_from:
  - /2018/05/14/
---
* Face verification chúng ta rất hay găp hằng ngày như facebook auto tag, hay face id của Iphone. .Hai model nổi tiếng hiện nay là 
facenet và deep face.Thằng facebook hình như nó xài deepface còn apple thì ko biết Trong model này mình sử dụng facenet với pre-training 
của openface.Mình đã tự thử build facenet với dataset(2k 12 class) nhưng kết quả ko cao. Model truyền thống là eigenface + svm accuracy chỉ có 60%.
Facenet trong project của mình cho acccuracy với svm là 94% còn knn là 87% tương đối cao.
Nói qua 1 tí về facenet. Data set nó chia ra 3 phần anchor(face gốc), Pog(face same face gốc) ,Neg(face different face gốc) .
Ý tưởng là nó sẽ maximun distance(anchor,Neg) và min distance(anchor,pog) sử dụng mạng CNN share weight và triplet loss để training.
Embbeding thành vector 128 dimension. Sau đó dùng feature này classifier dùng SVM,KNN. Để xác định face là stranger tức ko có trong 
database ta sử dụng 1 threshold làm ngưỡng.Model này mình dùng pre-traning của openface training trên 100k image với hơn 500 people.
Người ta sử dụng inception network thay cho simple CNN . Inception network có ưu điểm giúp mạng training sâu hơn bằng cách 2,3 layer 
convolution cùng 1 lúc sau đó concanation nó lại.
* Video demo : (https://www.youtube.com/watch?v=e00wwMXM9-Y&feature=youtu.be)
* link github : Bổ sung sau
