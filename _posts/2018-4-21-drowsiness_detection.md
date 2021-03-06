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
  
## Tìm hiểu ý tưởng .
* Ý tưởng cũng rất đơn giản thôi, là chúng ta sẽ dựa vào facial landmark của eyes để xác định được tỉ lệ nào đó như một ngưỡng để xem xét
mắt đang nhắm hay mở.Trong paper **Real-Time Eye Blink Detection using Facial Landmarks** của **Tereza Soukupova** và **Jan ´ Cech** đã
tìm ra được một công thức giải quyết vấn đề này có tên gọi là eye aspect ratio(EAR).Chúng ta cùng tìm hiểu qua về công thức này.

$$
EAR =  \frac{||p_{2} - p_{6}|| + ||p_{3} - p_{5}||}{||p_{1} - p_{4}||}
$$

   * Trong đó $p_{i}$ là lankmark point của eye, ký hiệu **|| ||** là khoảng cách euclide.
![drowsiness1](/assets/images/drowness1.jpg)
* Đồ thị EAR,trong đó p1,p2,p3,p4,p5,p6 là landmark point của eye(lưu ý ta sẽ ký hiệu bắt đầu bằng 0 thay vì bằng 1 trong model).Biểu đồ bên dưới là đồ thị của EAR . Khi mà eye ta thấy là EAR sẽ nằm dưới threshold 0.15 và bình thường của nó sẽ lớn hơn 0.25. 
* Đó là ý tưởng của bài toán.Ở đây có 1 số lưu ý là :
   * Có 2 eye nên ta sẽ lấy trung bình của 2 eye để lấy EAR
   * Để tránh trường hợp nháy mắt hay hay detection sai ta sẽ cho EAR một khoảng thời gian đủ lâu để xác nhận là drowsiness.
   * Threshold sẽ do ta chọn theo ý muốn ta ta thấy hợp lý.

## Xây dựng model .
* Trước hết ta xây dựng các hàm helper .
* Đầu tiên là hàm chuyển landmark point thành array . Vì mặc định nó rất khó xài.
~~~ ruby
def landmark_transform(landmarks):
    land_mark_array = []
    for i in landmarks:
        land_mark_array.append([int(i.x),int(i.y)])
    return np.array(land_mark_array)
~~~
* Tiếp theo là hàm tính EAR.
~~~ ruby
def calculate_distance(eye):
    assert len(eye)== 6
    p0,p1,p2,p3,p4,p5 = eye
    distance_p1_p5 = np.sqrt((p1[0]-p5[0])**2 + (p1[1]-p5[1])**2)
    distance_p2_p4 = np.sqrt((p2[0]-p4[0])**2 + (p2[1]-p4[1])**2)
    distance_p0_p3 = np.sqrt((p0[0]-p3[0])**2 + (p0[1]-p3[1])**2)
    EAR = (distance_p1_p5 + distance_p2_p4)/(2*distance_p0_p3)
    return EAR
~~~
* Tiếp theo là hàm vẽ contours cho eye để tiện theo dõi. Chúng ta dùng `convexhull` để xấp xỉ hình elip giống với eye.
~~~ ruby
def draw_contours(image,cnt):
    hull = cv2.convexHull(cnt)
    cv2.drawContours(image,[hull],-1,(0,255,0),2)
~~~ 
* Cuối cùng là hàm **alarm** (thông báo) khi drowsiness được phát hiện
~~~ ruby
def sound_alarm():
    playsound.playsound("sound.mp3")
~~~
* Bây giờ ta gộp các function helper đã tạo thành một model hoàn chỉnh.

~~~ ruby
path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predict_landmark = dlib.shape_predictor(path)

cap = cv2.VideoCapture(0)
total=0
alarm = False
while cap.isOpened() == True :
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects = detector(frame_gray,1)
    if len(rects) > 0 :
        for i in rects:
            cv2.rectangle(frame,(i.left(),i.top()),(i.right(),i.bottom()),(0,255,0),2)
            land_mark = predict_landmark(frame_gray,i)
            left_eye = landmark_transform(land_mark.parts()[36:42])
            right_eye = landmark_transform(land_mark.parts()[42:48])
            draw_contours(frame,left_eye)
            draw_contours(frame,right_eye)
            EAR_left,EAR_right = calculate_distance(left_eye),calculate_distance(right_eye)
            ear = np.round((EAR_left+EAR_right)/2,2)
            cv2.putText(frame, "EAR :" + str(ear) ,(200, 100),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 0), 2)
            if ear > 0.25 :
                total=0
                alarm=False
                cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255, 0 ), 2)
            else:
                total+=1
                print(total)
                if total>10:
                    if not alarm:
                        sound_alarm()
                        cv2.putText(frame, "drowsiness detect" ,(10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break               
~~~

* Giải thích code một tí .( lưu ý các tham số do mình chọn các bạn có thể điều chỉnh theo ý mình).
  * Ta sẽ không nhắc lại cách tìm facial landmark với faced detection với Dlib( bạn đọc có thể xem lại ở bài trước)
  * Các thông số ta đặt trong model : threshold của EAR là 0.25(nhỏ hơn sẽ xem là close eye)
  * Point landmark của eye: left_eye : 37-42,right_eye : 43-49
  * Ta sẽ đếm số lần eye close nếu nó vượt quá 10 thì sẽ có "alarm" qua biến là `total`
  * `detector` và `predict_landmark` dùng để detection face và landmark
  * Nếu `detector` thấy face thì ta sẽ tính` EAR_left` và `EAR_right` sau đó tính trung bình được `ear`
  * Cuối cùng xem xét điều kiện nếu total >10 thì sẽ `alarm`

## Test model.
* Ta sẽ test thử model. Vì máy mình cũ và webcame rất tối nên nhiều khi bị lag hoặc đứng hình.
[https://www.youtube.com/watch?v=oROrBeClnec]
<div class="x-frame video" data-video="https://www.youtube.com/watch?v=oROrBeClnec"> </div>

* Tham Khảo : http://hanzratech.in/, https://pyimagesearch.com, learnopencv.com
