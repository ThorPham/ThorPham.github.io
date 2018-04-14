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
Dữ liệu chúng ta cần chuẩn bị gồm 2 phần . Một là positive sample ( gọi tắt là pos) là data pedestrian và chúng ta gắn label cho nó là 1. Thứ hai là negative sample (Neg) là dữ liệu không chứa pedestrian bạn có thể lấy như background, car, house ... và ta gắn nhãn là -1.(lưu ý nếu training trong opecv thì nhãn gắn bắt buộc là 1 và -1 ).
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
* Dữ liệu của ta gồm có 924 image pos có chiều (128, 64, 3) và ta sẽ tạo (15x50) image neg có chiều (128, 64, 3)
![pedestrian](/assets/images/pedestian1.jpg)
# 2,Trích chọn đặc trưng 
Ta sẽ dùng hog để trích chọn đặc trưng

~~~ ruby
def hog_feature(image):
    feature_hog = hog(image,orientations=9,pixels_per_cell=(8,8),
    cells_per_block=(2,2),block_norm="L2")
    return feature_hog
    
#feature extraction for image pos    
X_pos = []
y_pos = []
for path in path_pos :
    im = io.imread(path,as_grey=True)
    im_feature = hog_feature(im)
    X_pos.append(im_feature)
    y_pos.append(1)
    
#feature extraction for image neg
X_neg = []
y_neg = []
w = 64
h = 128
for path in path_neg :
    im = io.imread(path,as_grey=True)
    for j in range(15):
        x = np.random.randint(0,im.shape[1]-w)
        y = np.random.randint(0,im.shape[0]-h)
        im_crop = im[y:y+h,x:x+w]
        im_feature = hog_feature(im_crop)
        X_neg.append(im_feature)
        y_neg.append(-1)
        
~~~
* Đầu tiên ta định nghĩa 1 function tính hog gồm các tham số `orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm="L2"`
* Sau đó tính hog trên pos và neg sample
* Cuối cùng ta stack pos và neg lại để chuẩn bị training
~~~ ruby
X_pos = np.array(X_pos)
X_neg = np.array(X_neg)
X_train = np.concatenate((X_pos,X_neg))
y_pos = np.array(y_pos)
y_neg = np.array(y_neg)
y_train = np.concatenate((y_pos,y_neg))
~~~
* Dữ liệu trining gồm có `X_traing` có shape (1674, 3780) gồm 1674 image và 3780 feature, `y_training` có shape là (1674,) gồm 2 giá trị 1 là pedestrian và -1 là non-pedestrian

# 3,Build model
Chúng ta sẽ training model bằng thuật toán svm có trong thư viện sklearn.
~~~ ruby
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
model = LinearSVC(C=0.01)
model.fit(X_train,y_train)
y_predict = model.predict(X_train)
print(classification_report(y_train,y_predict))
~~~
* Kết quả như sau :

![confustion_matrix](/assets/images/confustion_matrix.jpg)

* Amazing! kết quả accuracy = 100% . Quá cao phải ko. Nhưng đừng mừng vội vì data của chúng ta rất nhỏ và ta dùng toàn bộ data vào training mà ko chia ra data testing nên rất có thể bị overfiting. Khi đó model đưa vào hoạt động sẽ predict không tốt. Để tránh điều này
ta có thể thay đổi threshold để làm tăng precission ( vì khi predict trên image lớn sẽ có rất nhiều non-pedestrian và khi đó data của chúng ta sẽ unbalance )


