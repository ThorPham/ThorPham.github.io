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


![car](/assets/images/image1.jpg)
1, Nhận diện được vị trí của biển số xe trên image
Ý tưởng sẽ là cố gắng chỉ giữa lại những edge có khẳn năng là biển số xe nhất và loại bỏ những thứ không cần thiết khác.
~~~ ruby
im = cv2.imread("./car/IMG_0392.jpg")
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
noise_removal = cv2.bilateralFilter(im_gray,9,75,75)
equal_histogram = cv2.equalizeHist(noise_removal)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=20)
sub_morp_image = cv2.subtract(equal_histogram,morph_image)
ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
canny_image = cv2.Canny(thresh_image,250,255)
kernel = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
~~~
* load image cv2.imread
* Chuyển về ảnh xám cv2.cvtColor
* Remove noise bằng cv2.bilateralFilter.Bilateral filter khác với các filter khác là nó kết hợp cả domain filters(linear filter) và
 range filter(gaussian filter). Mục đích là giảm noise và tăng edge(làm egde thêm sắc nhọn edges sharp).
 * Cân bằng lại histogram cv2.equalizeHist làm cho ảnh ko quá sáng hoặc tối 
 * Morphogoly open ( open là erosion sau đó dilation) mục đích là giảm egde nhiễu , egde thật thêm sắc nhọn bằng cv2.morphologyEx sử dụng kerel 5x5
 * Xóa phông(background) không cần thiết bằng cv2.subtract(equal_histogram,morph_image)
 * Dùng threshold OTSU(làm việc rất tốt trong bimodel histogram) đưa ảnh về trắng đen tách biệt background và region interesting
 * Sử dụng thuật toán Canny để nhận biết egde bằng cv2.Canny
 * Cuối cùng dilate để tăng sharp cho egde
 
 ![car1](/assets/images/car1.jpg)
 
 ![car2](/assets/images/car2.jpg)
 
Đến đây ta thấy image của chúng ta đã tách được các egde ra khỏi image ban đầu. Để ý là biển số xe có hình chữ nhật nên ta sẽ dùng contour để lấy nó ra khỏi image.
~~~ ruby
new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * peri, True) 
    if len(approx) == 4:
            screenCnt = approx
            break
~~~
* Đầu tiên ta dùng cv2.findContours để tìm contour ( nó sẽ trả về 3 giá trị ta chỉ quan tâm giá trị thuws2 )
* Lọc contour theo area chỉ lấy 10 contour có giá trị lớn nhất( tránh lấy nhiều vì sẽ có nhiễu)
* Tiếp theo ta tính chu vi của từng contour bẳng cv2.arcLength sau đó dùng cv2.approxPolyDP để xấp xỉ đa giác ở đây ta cần tìm là hình chữ nhật nên ta chỉ giữ lại contour nào có 4 cạnh .
* Tách contour ra khỏi image ta thu được hình bên dưới
![plate](assets/images/plate.jpg)

