# Zeus - An Open Source Machine Learning Library

Zeus is created to provide an in-depth knowledge of some commonly used Machine Learning Algorithms.

## Prerequisites

- Numpy
- Python 3

You can install numpy using pip.

```
$ pip install numpy
```

## Installing Zeus

Download or clone the Zeus Repository from the following [Link](https://github.com/shibli2700/shibli2700.github.io)

Run the following command on your terminal to install Zeus

```
$ python setup.py build
```

## linear_model example

You can perform Linear Regression, Multiple Linear Regression using the linear_model package included in Zeus.

```python
from zeus.linear_regressors import regressors

regressor = regressors.LIRegressor()

x_train = [[1],[1.5],[3],[4.5]]
y_train = [[10000],[20000],[30000],[45000]]

regressor.train(x_train, y_train)
regressor.predict(5)
```

## Decision Tree Classifier Example

Decision Tree Classification can be done using the class DTreeClassifier() of the tree package.

```python
from zeus.tree import classifiers

classifier = classifiers.DTreeClassifier()

#training data having features such as Ear size, Body Size
training_data = [["Long","Big","Dog"],["Short","Small","Cat"],["Long","Small","Dog"]]

#test data
test_data = [["Short","Big"],["Big","Short"]]
node = classifier.train(training_data)

classifier.predict(test_data, node)
```
### Output
```
[{'Cat': '100%'}, {'Dog': '100%'}]
```
