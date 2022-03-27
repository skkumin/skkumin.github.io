---
layout: single
title:  "교차 검증(Cross Validation)"
categories: Python Machinelearning
tag: [Python, Machine learning, Cross Validation]
toc: true
toc_sticky: true
tagline: ""
header:
  overlay_image: /assets/images/ai.jpg
  overlay_filter: 0.5

---

### cross_validation을 하는 이유

머신러닝 모델을 구축할때 사용할 label 데이터가 있고 이 데이터를 test와 train data로 나누어서 사용한다면 고정된 test set을 가지고 모델의 성능을 확인하고 파라미터를 수정하는 과정을 반복하면 결국 고정된 test data에 overfitting이 이루어지는 현상이 나타난다. 이러한 문제를 해결하기 위해 cross_validation을 이용한다.  
모든데이터가 쓰이기 때문에 데이터의 수가 적을때도 사용한다. 

### k-fold cross validation(k-겹 교차 검증)
 - 가장 일반적으로 사용되는 cross validationn
 - k개의 fold를 만들어서 진행한다.
 - training/validation/test 3개의 집단으로 분류하는 것보다, training과  test로만 분류할 때 학습 데이터 셋이 더 많기 때문에 총데이터 갯수가적은 데이터 셋에 대하여 정확도를 향상시킬수 있다.
 - 가지고 있는 데이터의 100%를 테스트셋으로 사용할 수 있다.

### k-fold cross validation 과정
![](https://github.com/skkumin/skkumin.github.io/blob/master/images/dataminig/cross%20validation.png?raw=true){: width="60%" height="60%"}
 - 먼저 모든 데이터를 K개의 fold로 나눈다.
 - 그리고 그중에 k-1개의 fold로 trainind을 개의fold로 test를 수행한다.
 - 모든 fold가 한번씩 test를 수행할 수 있도록 전체 fold를 다 test에 사용하도록 K번 수행한다.
 - 각 과정마다 정확도를 계산하고 최정적으로 평균을 내어 정확도를 계산한다.

k-fold cross validation을 진행할 경우, 전체 데이터를 k개의 fold로 나누어서 진행하는 경우도 있고, 전체 데이터를 train 과 test로 나눈 후 train을 k개의 fold로 나누어 진행하는 경우가 있다.

### scikit-learn(`corss_val_score`) & k-fold cross validation
 - estimator: 평가하려는 모델
 - x: 훈련 데이터
 - y: 타깃 레이블
 - cv: 교차 검증 분할 수 (default:5)
 - 반환값: 교차 검증 결과 정확도 점수의 배열


```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
model = LogisticRegression()

scores = cross_val_score(model, iris.data, iris.target)

for i in range(5):
    print(f"{i+1}번째 정확도: {scores[i]}")
```

    1번째 정확도: 0.9666666666666667
    2번째 정확도: 1.0
    3번째 정확도: 0.9333333333333333
    4번째 정확도: 0.9666666666666667
    5번째 정확도: 1.0
    

    C:\Users\kevin\miniconda3\lib\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    
