---
layout: post
title: "Sentiment Analysis sử dụng Tf-Idf áp dụng cho ngôn ngữ tiếng việt "
description: "phân tích cảm xúc"
categories: [python,machine_learning]
tags: [python,machine learning]
redirect_from:
  - /2018/04/14/
---
Text mining ( lấy thông tin từ text) là một lĩnh vựng rộng và áp dụng trong nhiều lĩnh vực khác nhau. Một số ứng dụng có thể kể đến là :
sentiment analysis, document classification, topic classification, text summarization, machine translation. Trong bài hôm nay ta sẽ tìm
hiểu về sentiment analysis.Phân tích cảm xúc(sentiment analysis) được hiểu đơn giản là đánh giá 1 câu nói, tweet là tích cực (pos) hay tiêu
cưc(neg). Chẳng hạn lấy một ví dụ, bạn mở một cửa hàng bán đồ ăn mà muốn biết trên mạng xã hội người ta nói gì về quán ăn của bạn.Bạn bắt đầu
vào face, instagram hay tweeter để thu thập các commnent liên quan đến quán ăn của bạn. Bạn bắt đầu đoc thì có người khen người chê, vấn đề
xảy ra là bây giờ số comment nó tăng lên 1000 hay 10000 bạn có đủ sức đọc các comment đó hay không.Bạn bắt đầu nghĩ ra sẽ build một model làm
việc đó cho bạn
