---
layout: single
title:  "Neural Network 순전파 행렬곱셈 과 배치처리"
categories: DeepLearning
tag: [DeepLearning, propagatiom]
toc: true
toc_sticky: true
tagline: ""
header:
  overlay_image: /assets/images/ai.jpg
  overlay_filter: 0.5

---

### 순전파시 Numpy 행렬곱셉

![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/Propagationmatrix.jpg?raw=true){: width="60%" height="60%"}


```python
def sigmoid(x):
    return 1/(1+np.exp(-x))
```


```python
X = np.array([1, 2])
W = np.array([[0.1, 0.2], [0.3, 0.4]])
B = np.array([0.1, 0.2])
```


```python
Z = sigmoid(np.dot(X, W) + B)
print(Z)
```

    [0.68997448 0.76852478]
    

### 배치처리

사이킷 런의 iris 데이터는 feature의 수가 총 4개 그리고 타겟은 3가지의 class이다.  
이를 1번째층은 50개의 hidden layer, 2번째 층은 100개의 hidden layer를 가진 신경망 모델을 만든다고 하면 하나의 데이터에 대한 신경망 배열의 현상은 다음과 같다.  
                       
                   X      W1      W2      W3    ->    Y
                     
                   4     4X50   50X100   100X3        3

하지만 iris데이터의 총 수는 150개 이므로 이 데이터를 하나씩 계산하면 오래 걸린다. 이 계산시간을 줄이는 방법으로 배치처리를 이용한다.
  
                  X      W1      W2      W3    ->    Y
                  
                150X4   4X50   50X100   100X3      150X3
이렇게 되면 150X3의 행렬로 출력된다. 이처럼 하나로 몪은 입력 데이터를 배치라고 한다.
