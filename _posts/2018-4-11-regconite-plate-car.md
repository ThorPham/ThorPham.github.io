---
layout: post
title: "Nhận dạng biển số xe với opencv"
description: "nhận dạng biển số xe bằng image processing"
categories: [OpenCV]
tags: [python,opencv]
---
Nhận dạng biển số xe chắc không còn xa lạ đối với chúng ta, hàng ngày đi gửi xe ở các chung cư hay trung tâm thương mại chúng ta đều có thể nhìn thấy một anh bảo vệ ngồi gần một chiếc máy tính soi đi soi lại trên màn hình cái gì đó, đôi khi bảo chúng ta tắt đèn xe mà chúng ta chẳng hiểu để làm gì. Thực ra là có 1 camera ở phía sau chụp lại biển số xe của chúng ta. Anh ta đang xem lại ảnh trên máy tính có mờ hay nhiễu gì không để máy tính có thể nhận dạng được các con số trên biển số xe của chúng ta. Trong bài này chúng ta sẽ tìm hiểu cách mà máy tính có thể nhận dạng được các con số hay chữ cái. Có rất nhiều phương pháp và thuật toán có thể giải quyết được vấn đề này từ những thuật toán machine learning hay những thuật toán hiện đại hơn là CNN + RNN trong deep learning.
Các bước thực hiện :
#### 1, Nhận diện được vị trí của biển số xe trên image ( Object Localization)
#### 2, Segmentation các kí tự trên biển số xe
#### 3, Nhận dạng
Hai bước khó nhất là bước 1 và bước 2. Có một điểm chúng ta cần lưu ý là ở đây là camera đã đặt cố định và các character trên biển số xe
là tách biệt với nhau nên ta có thể dùng image procssing thông thường để lấy vị trí. Khác với những trường hợp nhận dạng realtime thì kỹ
thuật này không đạt được hiểu quả cao vì sẽ có rất nhiều nhiễu và image có rất nhiều hình thái khác nhau( rotation,scale..) nên khuyến nghị dùng deep learning sẽ hiểu quả hơn.

*Một số biển số xe lấy trên mạng .


![car](/assets/images/image1.jpg)
## 1, Nhận diện được vị trí của biển số xe trên image
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
* load image `cv2.imread`
* Chuyển về ảnh xám `cv2.cvtColor`
* Remove noise bằng `cv2.bilateralFilter.Bilateral` filter khác với các filter khác là nó kết hợp cả domain filters(linear filter) và
 range filter(gaussian filter). Mục đích là giảm noise và tăng edge(làm egde thêm sắc nhọn edges sharp).
 * Cân bằng lại histogram `cv2.equalizeHist` làm cho ảnh ko quá sáng hoặc tối 
 * Morphogoly open ( open là erosion sau đó dilation) mục đích là giảm egde nhiễu , egde thật thêm sắc nhọn bằng cv2.morphologyEx sử dụng kerel 5x5
 * Xóa phông(background) không cần thiết bằng `cv2.subtract(equal_histogram,morph_image)`
 * Dùng threshold `OTSU`(làm việc rất tốt trong bimodel histogram) đưa ảnh về trắng đen tách biệt background và region interesting
 * Sử dụng thuật toán Canny để nhận biết egde bằng `cv2.Canny`
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
* Đầu tiên ta dùng `cv2.findContours` để tìm contour ( nó sẽ trả về 3 giá trị ta chỉ quan tâm giá trị thứ 2 )
* Lọc contour theo area chỉ lấy 10 contour có giá trị lớn nhất( tránh lấy nhiều vì sẽ có nhiễu)
* Tiếp theo ta tính chu vi của từng contour bẳng cv2.arcLength sau đó dùng `cv2.approxPolyDP` để xấp xỉ đa giác ở đây ta cần tìm là hình chữ nhật nên ta chỉ giữ lại contour nào có 4 cạnh .
* Tách contour ra khỏi image ta thu được hình bên dưới

![plate](/assets/images/plate.jpg)

## 2, Segmentation các kí tự trên biển số xe:
Bước này đơn giản hơn bước 1 .Ý tưởng là lọc nhiều sau đó dùng contour để tách các character ra khỏi image.
~~~ ruby
roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
roi_blur = cv2.GaussianBlur(roi_gray,(3,3),1)
ret,thre = cv2.threshold(roi_blur,120,255,cv2.THRESH_BINARY_INV)
kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
thre_mor = cv2.morphologyEx(thre,cv2.MORPH_DILATE,kerel3)
_,cont,hier = cv2.findContours(thre_mor,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
~~~
* Tương tự như bước 1 ta sẽ lọc nhiễu bằng cv2.GaussianBlur
* Dùng cv2.THRESH_BINARY_INV đưa ảnh về trắng đen
* Dùng cv2.MORPH_DILATE 
* Cuối cùng tìm contour trên image. Kết quả như sau

![contour](/assets/images/contour.jpg)

* Để ý sẽ có 7 charater trên plate mà ta cần lấy mà lại có một số contour nhiễu nên ta sẽ tính area của contour sau đó sorted và bỏ 2 contour đầu(vì ta dùng mode `cv2.RETR_LIST` nên sẽ có 1 contour bao toàn bộ image và 1 contour bao đường biên plate) và lấy 7 area lớn nhất.
~~~ ruby
areas_ind = {}
areas = []
for ind,cnt in enumerate(cont) :
    area = cv2.contourArea(cnt)
    areas_ind[area] = ind
    areas.append(area)
 areas = sorted(areas,reverse=True)[2:9]
 for i in areas:
    (x,y,w,h) = cv2.boundingRect(cont[areas_ind[i]])
    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
~~~
![final](/assets/images/final.jpg)

## 3 : Nhận dạng : 
Đến bước này chúng ta có thể dùng machine learning hoặc temple machine để nhận dạng các character. Nếu các bạn muốn tìm hiểu thêm vui lòng đọc lại bài viết nhận dạng chữ số viết tay.

## Kết luận
Chúng ta chỉ mới sử dụng image processing thuần túy để localizer and segmentation mà chưa dùng bất kỳ thuật toán nào cao siêu cả. Thế mới thấy image processing rất quan trọng trong lĩnh vực computer vision. Cách làm này chỉ áp dụng tốt khi camera đặt cố định khi đó ta có thể tinh chỉnh một số hàm để cho nó phù hợp với các trường hợp khác nhau. Tuy vậy nó cũng làm việc không tốt khi car có màu trắng trùng với màu của biến số. 
### Tham khảo :
https://www.pyimagesearch.com/
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html
https://learndeltax.blogspot.com
