---
layout: post
published: True

title: "Facial landmark với Dlib "
description: "tìm hiểu facial landmark"
categories: [python,computer_vision]
tags: [python,computer_vision]
---
## Mở đầu .
Facial landmark là xác định các vị trí như eye(mắt), nose(mũi),mounth(miệng) trên khuôn mặt(face). Nó có rất nhiều ứng dụng vui mà ta thường thây
trên các app điện thoại chẳng hạn ( swap face,draw in face or tạo hiệu ứng trên khuôn mặt). Trong bài này chúng ta sẽ tìm hiểu về thư viện
Dlib trong python để xác định Facial landmark.
* Mục lục
  * Tìm hiểu về Dlib
  * Face detection với Dlib
  * Face landmark với Dlib
  
## Tìm hiểu về Dlib
Thư viện Dlib được viết bằng ngôn ngữ C++ do **Davis King**  tạo ra vào năm 2012. Được sử dụng nhiều trong lĩnh vực computer vision đặc biệt là nhận dạng object và face. 
## Face detection với Dlib

* Cũng giống như **Opencv** , Dlib cũng hỗ trợ nhận dạng khuôn mặt. Dlib sử dụng hog(histogram oriented gradient) làm feature và training bằng linear classifer kết hợp với image pyramid và window search ( bạn đọc có thể tìm hiểu thêm ở các bài trước).
* Để detectoion face trong dlib trước tiên ta cần tạo một object `detector = dlib.get_frontal_face_detector()`. object `detector` có 2 tham số là chúng ta quan tâm 
  * Một là image chúng ta muốn detect
  * Hai là tham số upsample . Là một `int` khi mà chúng ta muốn detection các object nhỏ hơn. Thì tham số này có tác dụng phóng to ảnh.
~~~ ruby
import numpy as np
import dlib
import cv2
#create object detection
detector = dlib.get_frontal_face_detector()
# load image
im = cv2.imread("em_thuy.jpg")
rects = detector(im,1)
# xem kết quả
for d in rects:
    cv2.rectangle(im,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),2)
    cv2.imshow("im",im)
    cv2.waitKey()
    cv2.destroyAllWindows()
~~~
* Giải thích code
  * Tạo object detection với function `get_frontal_face_detector`
  * Sau đó Load image và detection
  * Kết quả trả về sẽ là 1 list các face detection được. Mỗi face detection là 1 rectangle tuple theo cấu trúc (left,top),(right,bottom).
  * lưu ý có thể detection nhiều face trên một image.
  
![face](/assets/images/dlib1.jpg)

## Face landmark với Dlib
* Facical landmark có cầu trúc như hình bên dưới. Dlib sẽ detection 62 point trên khuôn mặt.
  * Mouth point = 48-61
  * Right_brow_point = 17-21
  * Left_brow_point = 22-26
  * Right_eye_point = 36-42
  * Left_eye_pint = 42-48
  * Nose_point = 27-35
  * Jaw_point = 0-17
![landmark](/assets/images/facial_landmark.jpg)
~~~ ruby
path = "shape_predictor_68_face_landmarks.dat"
predict = dlib.shape_predictor(path)
landmark = predict(im,rects[0])
for idx,point in enumerate(landmark.parts()):
    cv2.circle(im,(point.x,point.y),1,(0,255,0),1)
    cv2.putText(im,str(idx),(point.x,point.y),fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
cv2.imshow("im",im)
cv2.waitKey()
cv2.destroyAllWindows()
~~~
![landmark](/assets/images/facial_landmark1.jpg)
