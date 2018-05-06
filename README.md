# Zeus - An Open Source Machine Learning Library

Zeus is created to provide an in-depth knowledge of some commonly used Machine Learning Algorithms.

## Prerequisites

What things you need to install to run Zeus

```
numpy
```

## linear_model example

You can perform Linear Regression, Multiple Linear Regression using the linear_model package included in Zeus.

```
from zeus.linear_regressors import regressors

regressor = regressors.LIRegressor()

x_train = [[1],[1.5],[3],[4.5]]
y_train = [[10000],[20000],[30000],[45000]]

regressor.train(x_train, y_train)
regressor.predict(5)
```
