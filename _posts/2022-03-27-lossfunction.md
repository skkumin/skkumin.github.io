---
layout: single
title:  "Neural Network 손실함수(Loss funstion, Cost function) "
categories: DeepLearning
tag: [DeepLearning, loss funstion]
toc: true
toc_sticky: true
tagline: ""
header:
  overlay_image: /assets/images/ai.jpg
  overlay_filter: 0.5

---

### 손실 함수란?
 - 신경망은 "하나의 지표"를 기준으로 최적의 매개변수 값을 탐색한다. 신경망 학습에서 사용하는 지표를 손실함수라고 한다.
 - 손실 함수는 신경망 성능의 "나쁨"을 나타내는 지표로, 현재의 신경망이 훈련 데이터를 얼마나 잘 처리하지 "못"하느냐를 나타낸다.
 - 일반적으로 손실함수로 (교차 엔트로피) 와 (오차제곱합)을 사용한다.

### 왜 손실함수를 이용하는가?
"정확도"라는 지표를 나두고 "손실함수"라는 우회적인 방법을 선택하는 이유는 바로 신경망 학습에서의 "미분"때문이다. 신경망 학습에서는 최적의 매개변수를 탐색할 때 손실 함수의 값을 가능환 한 작게 하는 매개변수 값을 찾는다. 이때 매개변수의 미분을 계산하고, 그 미분값을 단서로 매개 변수의 값을 서서히 갱신하는 과정을 반복한다.  
미분값이 음수면 가중치 매개변수를 양의 방향으로 변환시켜 손실 함수의 값을 줄일 수 있고, 반대로 미분 값이 양수이면 가중치 매개변수를 음의 방향으로 변환시켜 손실 함수의 값을 줄일 수 있다.

### 오차 제곱 합(sum of squares for error, SSE)
 - regression문제의 손실함수로 사용  
 ![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/sse.png?raw=true){: width="60%" height="60%"}
 - y_k: 신경망의 출력(신경망이 추정한 값)
 - t_k: 실제값
 - k: 데이터의 차원의 수


```python
def get_SSE(y_k, t_k):
    error = np.sum(np.square(y_k - t_k))/2
    
    return error
```


```python
y_k = np.array([1, 3, 5, 6, 7])
t_k = np.array([3, 2, 7, 5, 8])

get_SSE(y_k, t_k)
```




    5.5



### 교차 엔트로피 오차(cross entropy error, CEE)
 - classification문제의 손실 함수로 사용  
 ![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/CEE.png?raw=true){: width="60%" height="60%"}

 - log: 밑이 e인 자연로그
 - y_k: 신경망의 출력
 - t_k: 정답 레이블
 - t_k 는 정답에 해당하는 인덱스의 원소만 1이고 나머지는 0이다.(원핫 인코딩) 그래서 실질적으로는 t_k가 1일 때의 y_k의 자연로그를 계산하는 식이 된다. 즉, 교차 엔트로피 오차는 정답일 때의 출력이 전체 값을 정하게 된다.


```python
def get_CEE(y_k, t_k):
    error = -np.log(np.sum(y_k*t_k))
    
    return error
```


```python
y_k = np.array([0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0])
t_k = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

get_CEE(y_k, t_k)
```




    0.5108256237659907


