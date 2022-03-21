---
layout: single
title:  "Decision Tree with Python"
categories: Python Machinelearning
tag: [Python, Machine learning, Classification, DecisionTree]
toc: true
toc_sticky: true
tagline: ""
header:
  overlay_image: /assets/images/ai.jpg
  overlay_filter: 0.5

---

```python
from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
```

### iris 데이터셋 사용


```python
iris=load_iris()
```


```python
iris.feature_names
```




    ['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']




```python
iris.target_names
```




    array(['setosa', 'versicolor', 'virginica'], dtype='<U10')



### Decision Tree 구축 및 plot_tree를 이용한 시각화


```python
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```


```python
model1 = DecisionTreeClassifier(criterion="entropy", max_depth=None) #깊이를 제안하는 파라미터 : max_depth
model.fit(x_train, y_train)
```




    DecisionTreeClassifier(criterion='entropy')




```python
plt.figure(figsize=(12, 12))
plot_tree(decision_tree=model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```


    
![png](output_8_0.png)
    


![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/ptree1.png?raw=true)

### Decision Tree의 단점
아래는 randomstate=2인 train data와 test data를 같은 비율이지만 다르게 나눈 동일한코드이다.  
아래의 도식화된 그림을 보면 학습데이터에 변동이 최종결과에 큰 영향을 주는것을 알 수 있다.  


```python
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model2 = DecisionTreeClassifier(criterion="entropy", max_depth=None) #깊이를 제안하는 파라미터 : max_depth
model2.fit(x_train, y_train)

plt.figure(figsize=(12, 12))
plot_tree(decision_tree=model2, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```


    
![png](output_11_0.png)
    


![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/ptree2.png?raw=true)

### Decision Tree Prunning
Decision Tree Clasifier의 max_depth의 조정으로 1 ~ max일때까지 test error를 통해 test error가 가장적은 노드를 기준으로 Prunning을 진행하였다. 


```python
max_depths = [depth+1 for depth in range(int(model.get_depth()))]
max_depths
```




    [1, 2, 3, 4]




```python
train_error = []
test_error = []
for depth in max_depths:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(x_train, y_train)
  
    target_train_predict = model.predict(x_train)
    target_test_predict = model.predict(x_test)

    acc_train = accuracy_score(y_train, target_train_predict)
    acc_test = accuracy_score(y_test, target_test_predict)
    train_error.append(1-acc_train)
    test_error.append(1-acc_test)  
```


```python
plt.plot(max_depths, train_error, color="red")
plt.plot(max_depths, test_error, color="blue")

plt.show()
```


    
![png](output_16_0.png)
    


![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/ptree3.png?raw=true){: width="60%" height="60%"}

위의 graph를 통해 파란색 test error을 보면 depth가 3일때 Prunning을 해야하는것을 알 수 있다.


```python
model_f = DecisionTreeClassifier(criterion="entropy", max_depth=3) 
model_f.fit(x_train, y_train)
```




    DecisionTreeClassifier(criterion='entropy', max_depth=3)




```python
plt.figure(figsize=(12, 12))
plot_tree(decision_tree=model_f, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```


    
![png](output_20_0.png)
    


![](https://github.com/skkumin/skkumin.github.io/blob/master/images/deeplearning/ptree4.png?raw=true)

위의 Decision Tree가 Pruning을 끝낸 최종 결과이다.
