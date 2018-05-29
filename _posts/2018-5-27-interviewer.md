---
layout: post
title: "Một số câu hỏi interviewer AI,Deep learning,machine learning"
description: "Một số câu hỏi interviewer AI,Deep learning,machine learning"
categories: [computer_vision]
tags: [computer_vision,python]
redirect_from:
  - /2018/04/22/
---
* 1, Why is naive bayes called “naive” ?.
  * Tại vì trong naive bayes ta đã giải định các feature không có mối quan hệ với nhau,tức là xác suất xảy ra của feature này không bị ảnh
  hưởng bởi feature kia. Lấy vd : text = " bóng đá được xem là môn thể thao vua" thì xác xuất cụm từ "thể thao" xuất hiện không bị ảnh hưởng bởi
  cụm từ "bóng đá". Trên thực tế điều này không đúng ha vì 2 cụm từ này có mối quan hệ tương đối mật thiết với nhau. Đây cũng là một nhược điểm
  của thuật toán này.
* 2, Tell me about naive bayes classifier ?
  * Naive bayes classifier dựa trên công thức bayes: $ P(a/b) = \frac{P(a\cap b)P(a)}{P(b)}$.Hiểu một cách đơn giản hơn
  $ posterior = \frac{likehood x prior}{evident} $
  * Giả sử có 2 class (c1,c2), thì hiểu một cách đơn giản naive bayes là ta tìm xác suất của $P(c1/b)$,$P(c2/b)$ rồi so sánh 2 xác suất này xem cái nào lớn hơn thì sẽ thuộc về class đó. Vì P(b) là như nhau nên ta chỉ cần tính $P(a\cap b)$ và $P(a)$
* 3, Explain TF-IDF ?
  * TF-IDF nó là viết tắt của từ Term frequency invert document frequency.Nó là một kỹ thuật feature extraction dùng trong text mining và information retrieval. Trước khi có tf-idf người ta dùng one-hot-encoding để embedding words sang vector. Nhưng kỹ thuật này gặp một số hạn chế là :
  * Những từ thường xuyên xuất hiện sẽ không có nhiều thông tin nhưng vẫn có tỉ trọng(weight) ngang với các từ khác.vd : stop word chẳng hạn hay chúng ta phân tích vềquán ăn nào đó thì từ "quán ăn" xuất hiện ở tất cả document.Chúng ta cần giảm tỉ trọng về mặt thông tin nó  xuống vì thông tin không mang nhiều giá trị.
  * Những từ hiếm(rare word) or key word không có sự khác biệt về tỉ trọng thông tin
  * Để khắc phục hạn chế này tf-idf đã ra đời.Tf-idf bao gồm 2 thành phần là tf(term frequency) và idf(inverse document frequency)
<div style="text-align: center"> $$
tf(w,d) = \frac{\text{number of word w in document d}}{\text{total word in document}}
$$ </div>
  * tf đo lường tỉ trọng tần suất từ w có trong document d.Vì document thường có lenght khác nhau nên để normalization ta chia nó cho number word trong document d.
<div style="text-align: center"> $$
idf = tf* \frac{N}{\text{documnet in word w appear}}
$$ </div>
  * N là tổng số document trong dataset.Tỉ số $\frac{N}{\text{documnet in word w appear}}$ được xem là inverse document frequency. Nếu một từ xuất hiện nhiều ở các document thì tỉ số này sẽ gần 1.Và ngược lại một từ ít xuất hiện hơn tỉ số này sẽ cao hơn 1. Điều này giúp giảm tỉ trọng của 
những từ thường xuyên suất hiện và tăng tỉ trọng những từ ít xuất hiện trong document hơn (lưu ý N luôn lớn hơn hoặc bằng documnet in word w appear).
  * Một vấn đề là khi N rất lớn mà `documnet in word w appear` rất nhỏ thì tỉ số này rất lơn cho nên là người dùng log transform để giảm giá trị tỉ số $\frac{N}{documnet in word w appear}$ tránh gây khó khăn trong việc tính toán ( lưu ý log nó làm giảm giá trị theo cấp lũy thừa). Khi đó công thức idf cuối cùng sẽ là 
<div style="text-align: center "> $$
idf = tf* log(\frac{N}{\text{documnet in word w appear}})
$$ </div>
  * Ví dụ : Một document 100 word chứa word cat 3 lần. $ tf = \frac{3}{100} = 0.03 $ . Giả sử có 10000 document mà word cat xuất hiện trong 1000 document. $ idf(cat) = 0.03* log(\frac{10000}{1000}) = 0.06 $
* 4, What are word2vec vectors?
  * Như chúng ta đã biết máy tính được cấu tạo từ những con số, do đó nó chỉ có thể đọc được dữ liệu số mà thôi. Trong natural language processing thì để xử lý dữ liệu text chúng ta cũng phải chuyển dữ liệu từ text sang numeric, tức là đưa nó vào một không gian mới người ta thường gọi là embbding. Trước đây người ta mã hóa theo kiểu one hot encoding tức là tạo một vocabualary cho dữ liệu và mã hóa các word trong document thành những vectoc, nếu word đó có trong document thì mã hóa là 1 còn không có sẽ là 0. Kết quả tạo ra một sparse matrix, tức là matrix hầu hết là 0.Các mã hóa này có nhiều nhược điểm đó là thứ nhất là số chiều của nó rất lớn (NxM, N là số document còn M là số vocabulary), thứ 2 các word không có quan hệ với nhau. Điều đó dẫn đến người ta nghĩ ra một model mới có tên là Word embbding, ở đó các word sẽ có quan hệ với nhau về semantic tức là ví dụ như paris-tokyo,man-women,boy-girl những cặp từ này sẽ có khoảng cách gần nhau hơn trong Word embbding space. Ví dụ điển hình mà ta thây đó là phương trình king - queen = man - women . Cái ưu điểm thứ 2 là số chiều của nó sẽ giảm chỉ còn NxD. Word embbding có 2 model nổi tiếng là word2vec và Glove.
  * Word2vec được tạo ra năm 2013 bởi một kỹ sư ở google có tên là Tomas Mikolov. Nó là một model unsupervised learning,được training từ  large corpus. Word2vec có 2 model là skip-gram và Cbow.
    * Skip-gram model là model predict word surrounding khi cho một từ cho trước, ví dụ như text = "I love you so much". Khi dùng 1 window search có size 3 ta thu được : {(i,you),love},{(love,so),you},{(you,much),so}. Nhiệm vụ của nó là khi cho 1 từ center ví dụ là love thì phải predict các từ xung quang là i, you.
    * Cbow là viết tắt của continous bag of word . Model này ngược với model skip-gram tức là cho những từ surrounding predict word current.
    * Trong thực tế người ta chỉ chọn một trong 2 model để training, Cbow thì training nhanh hơn nhưng độ chính xác không cao bằng skip-gram và ngược lại
* 5, How does SVM learns non-linear boundaries ? Explain.
  * Khi dữ liệu non-linear thì không tìm được 1 mặt phẳng nào tối ưu để classifier nên người ta nghĩ ra dùng 1 kernel như 1 function map
  data ban đầu sang không gian mới mà ở đây có thể tìm được mặt phẳng tối ưu để classifier
* 6, What is precision and recall ? Which one of this do you think is important in medical diagnosis ?
  * Giả sử chúng ta có một cái máy chuẩn đoán bệnh cancer như hình dưới :
  ![recal](/assets/images/recall.jpg)
    * Precision được định nghĩa là : $ precision = \frac{TP}{TP + FP} $. Tỉ lệ máy chuẩn đoán số người bị bệnh ung thư trên thực tế người đó bị ung thư thật sự. Hiểu đơn giản hơn là accuracy của cái máy đoán bệnh ung thư. $precision = \frac{10}{15} = 66% $
    * Recall  được định nghĩa là : $ Recall = \frac{TP}{TP + FN} $ Tỉ lệ người bị cancer được máy chuẩn đoán đúng trên số người bị bệnh cancer. Hiểu đơn giản là accuracy của cái máy chỉ xét trên số người thực sự bị cancer : $precision = \frac{10}{14} = 71% $
* 7, Define precision and recall ?
* 8, What is random about Random Forest ?
  * Chúng ta biết random forest là một thuật toán ensemble method. Nó kết hợp nhiều thuật decision tree lại thành 1 rừng(forest). Random trong random forest có nghĩa là mỗi lần lấy data cho decision tree nó sẽ random một sample từ data training, random có thể là sample instance hoặc feature, và cũng có thể cả 2.
* 9, What are the criteria for splitting at a node in decision trees ? 
  * Một số criteria để plit node trong decission tree là :
    * Gini impurity : $ \sum_k{\neg 1}(p_{k} = 1 - p_{i} $
    * Entropy : $ -\sum_{i=1}^{J}p_{i}log(p_{i}) $
* 10, What is the advantage with random forest ?
  * Có 2 advantage đó là :
    * Giảm overfiting : 
    * Giảm variance :
* 11, Tell me about boosting algorithms ?
* 12, How does gradient boosting works ?
* 13, What are the kernels used in SVM ? What is the optimization technique of SVM ?
* 14, How do you decide K in K-Means clustering algorithm ?
* 15, Can you tell DB-SCAN algorithm ?
* 16, How does HAC (Hierarchical Agglomerative clustering) work ?
* 17, Explain PCA ? Tell me the mathematical steps to implement PCA ?
  * PCA là một trong những phương pháp giảm chiều dữ liệu ( Dimensionality reduction techniques ) phổ biến và được sử dụng trong nhiều lĩnh vực khác nhau. PCA có nhiều ứng dụng như tìm mối tương quan giữa các biến ( relationship between observation), trích xuất những thông tin quan trọng từ data, phát hiện và loại bỏ outlier và giảm chiều chiều dữ liệu.Ý tưởng của phương pháp PCA là tìm ra một không gian mới để chiếu(project) data sao cho variation giữ lại là nhiều nhất.
  *  Có 2 phương pháp tiếp cận PCA là covarian matrix và SVD .
  * Phương pháp Covarian matrix : Các bước thực hiện thuật toán như sau :
 ![pca](/assets/images/pca.jpg)
 
    * X data có chiều MxN ( với N là số sample ,M là số feature).
    * Tính mean của X :
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
    * Tính toán EigenVector **V** và EigenValue $\lambda$ của Covarian $\sum$
    * Sort EigenValue tương ứng với EigenVector theo thứ tự $\lambda$ giảm dần .
    * Chọn những EigenVector tương ứng với EigenValue lớn nhất $ W = \{v_{1},v_{2},..v_{k}\} $ . EigenVector W sẽ làm đại diện để project X vào PCA space
    * Tất cả sample X sẽ được project vào không gian nhỏ hơn theo công thưc $Y = W^{T}\cdot D$
* 18, What is disadvantage of using PCA ?
  * PCA chỉ xem xét trên global data không quan tâm đến local data( ngược lại với Tsne). Nó chỉ có một số limit trong giả định khi thực hiện thuật toán là :
    * Linearity : PCA giải định các principle components là linear combination
    * Large variance implies more structure : PCA sử dụng variace làm thước đo cho principle components, high variace thì principle components càng quan trong, low variace được xem như noise.
    * Orthogonality : Các principle components là Orthogonality với nhau.
* 19, How does CNN work ? Explain the implementation details ?
* 20, What is the range of sigmoid function ?
* 21, What is mean and variance of standard normal distribution ?
* 22, Which model would you use in case of unbalanced dataset: Random Forest or Boosting ? Why ?
* 23, What are Lasso and Ridge regression ?
* 24, What is Gaussian Mixture model ? How does it perform clustering ?
* 25, How is Expectation Maximization performed ? Explain both the steps ?
* 26, Explain the intuition behind BIC or AIC ?
* 27, What’s the trade-off between bias and variance?
* 28, What is the difference between supervised and unsupervised machine learning?
* 29, How is KNN different from k-means clustering?
* 30, Explain how a ROC curve works.
* 31, Explain the difference between L1 and L2 regularization.
* 32, What’s the difference between Type I and Type II error?
* 33, What’s the difference between probability and likelihood?
  * Hiểu một cách đơn giản probability function $ F(x\\theta)$ khi bạn biết tham số của hàm probability function thì bạn có thể xác định được tất cả các observation trên phân phối đó ( vd phân phôi chuẩn được qui định bởi 2 tham số là mu và sigma (3,0.4)). Còn hàm likehood
  $ F(\theta\x)$ thì ngược lại khi bạn có một số quan sát nhưng bạn chưa biết được tham số của hàm phân phối xác xuất bạn muốn ước lượng các tham số này.
* 34, What’s the difference between a generative and discriminative model?
* 35, How is a decision tree pruned?
* 36, Which is more important to your model accuracy, or model performance?
* 37, What’s the F1 score? How would you use it?
* 37,Explain max un-pooling operation for increasing the resolution of feature maps.
* 38,What is a Learnable up-sampling or Transpose convolution ?
* 39,Describe the transition between R-CNN, Fast R-CNN and Faster RCNN for object detection.
* 40,Describe how RPNs are trained for prediction of region proposals in Faster R-CNN?
* 41,Describe the approach in SSD and YOLO for object detection. How these approaches differ from Faster-RCNN. When will you use one over the other?
* 42,Difference between Inception v3 and v4. How does Inception Resnet compare with V4.
* 43,Explain main ideas behind ResNet? Why would you try ResNet over other architectures?
* 44, Explain batch gradient descent, stochastic gradient descent and mini-match gradient descent.
* 45, Explain Dropout and Batch Normalization. Why BN helps in faster convergence?
* 46, What is the difference between K-means and EM?
* 47, What is an auto-encoder?
* 48, What is exploding gradients ?
* 49, What is a confusion matrix ?
* 50, What is an imbalanced dataset? How is it handled?
