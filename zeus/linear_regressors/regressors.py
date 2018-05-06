"""
author : Mohammed Shibli
Date: 14-04-2018

"""
import numpy as np

class LIRegressor:
    
    sum_dependent = []
    sum_xn_and_yn = []
    sum_xn_and_xn = []
    rows = None
    cols = None
    
    def __init__(self):
        LIRegressor.sum_dependent = []
        LIRegressor.sum_xn_and_xn = []
        LIRegressor.sum_xn_and_yn = []
        LIRegressor.rows = None
        LIRegressor.cols = None
    
    def calculate_total(self,x_independent):
        sum_of_array = 0
        for x in np.nditer(x_independent):
            sum_of_array += x
            
        return float(sum_of_array)
    
    def calculate_slopes(self,x_train,y_train):
       
       LIRegressor.rows = np.shape(x_train)[0]
       LIRegressor.cols = np.shape(x_train)[1]
       #calculating summation x^n
       LIRegressor.sum_dependent.append(LIRegressor.rows)
       for col in range(0,LIRegressor.cols):
           LIRegressor.sum_dependent.append(self.calculate_total(x_train[:,col]))
           
       #calculating summation of x^n y
       LIRegressor.sum_xn_and_yn.append(self.calculate_total(y_train))
       for col in range(0,LIRegressor.cols):
           _sum = 0
           for row in range(0,LIRegressor.rows):
               _x = y_train[row] * x_train[row,col]
               _sum += _x
           LIRegressor.sum_xn_and_yn.append(_sum)
       
       LIRegressor.sum_xn_and_xn.append(LIRegressor.sum_dependent)
       for col in range(0,LIRegressor.cols):
           _dummy = []
           _dummy.append(self.calculate_total(x_train[:,col]))
           for col_2 in range(0,LIRegressor.cols):
               _sum = 0
               for row in range(0,LIRegressor.rows):
                   _x = x_train[row,col] * x_train[row,col_2]
                   _sum += _x
               _dummy.append(_sum)
           LIRegressor.sum_xn_and_xn.append(_dummy)
       try:
           slopes = np.linalg.solve(np.array(LIRegressor.sum_xn_and_xn),np.array(LIRegressor.sum_xn_and_yn))
       except Exception:
           slopes = np.linalg.lstsq(np.array(LIRegressor.sum_xn_and_xn),np.array(LIRegressor.sum_xn_and_yn), rcond=None)[0]
           
       return slopes
   
    def train(self,x_train,y_train):
        if x_train is None or y_train is None:
            raise Exception("Empty training data entered.")
        x_train = np.matrix(x_train)
        y_train = np.matrix(y_train)
        self.slopes = self.calculate_slopes(x_train,y_train)
    
    def predict(self,x_test):
        x_test = np.matrix(x_test)
        _predict = []
        _rows = np.shape(x_test)[0]
        _cols = np.shape(x_test)[1]
        if _cols == LIRegressor.cols:
            for row in range(0,_rows):
                prediction = self.slopes[0]
                _x = 1
                for col in range(LIRegressor.cols):
                    prediction = prediction + (x_test[row,col] * self.slopes[_x])
                    _x += 1
                _predict.append(prediction)
        else:
            raise Exception("Test data number of columns does not match with train data.")
        return np.array(_predict)