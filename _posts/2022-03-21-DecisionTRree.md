---
layout: single
title:  "[Decision Tree] 이론"
categories: Python Machinelearning
tag: [Python, Machine learning, Classification, DecisionTree]
toc: true
toc_sticky: true
tagline: ""
header:
  overlay_image: /assets/images/ai.jpg
  overlay_filter: 0.5

---

# 의사결정나무란?
![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/Tree_1.jpg?raw=true){: width="80%" height="80%"}

# 가지치기(분기) 의 기준을 정하는법 - Cross Entropy
![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/Tree_2.jpg?raw=true){: width="80%" height="80%"}

# Information Gain를 통한 가지치기의 예시
![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/Tree_3.jpg?raw=true){: width="80%" height="80%"}
![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/Tree_4.jpg?raw=true){: width="80%" height="80%"}
![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/Tree_5.jpg?raw=true){: width="80%" height="80%"}
![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/Tree_6.jpg?raw=true){: width="80%" height="80%"}

# 의사결정나무의 단점
 - 계층적 구조로 인해 중간에 에러가 발생하면 다음 단계로 에러가 계속 전파
 - 학습 데이터의 미세한 변동에도 최종 결과에 크게 영향
 - 적은 개수의 노이즈에도 크게 영향
 - 나무의 최종노드 개수를 늘리면 과적합 위험
