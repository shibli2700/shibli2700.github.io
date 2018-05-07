# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:28:10 2018

@author: Mohammed Shibli
"""

import numpy as np
 
class Leaf:
    def __init__(self, rows):
        self.predictions = DTreeClassifier.class_counts(self,rows)
        
class DecisionNode:
    def __init__(self,question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class DTreeClassifier:
    
    def getUnique(self,x):
        _unique = []
        for y in np.nditer(x):
            if y not in _unique:
                _unique.append(y)
                
        return np.array(_unique)
    
    def class_counts(self,traindata):
        counts = {}
        for row in traindata:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
            
            
        return counts
    
    def is_numeric(self,value):
        if(value.isdigit()):
            return True
        else:
            return False
    
    def gini(self,rows):
        counts = self.class_counts(rows)
        impurity = 1
        for label in counts:
            prob_of_label = counts[label] / float(len(rows))
            impurity -= prob_of_label ** 2
        return impurity
    
    def information_gain(self, left, right, uncertainity):
        p = float(len(left)) / (len(left) + len(right))
        return uncertainity - p * self.gini(left) - (1-p) * self.gini(right)
    
    def partition(self,rows, question):
        
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
                
        return np.array(true_rows), np.array(false_rows)
    
    def split_data(self,rows):
        best_gain = 0
        best_question = None
        uncertainity = self.gini(rows)
        features = len(rows[0]) - 1
        
        for col in range(0, features):
            values = self.getUnique(rows[:,col])
            for val in values:
                
                question = Question(col, val)
                true_rows, false_rows = question.partition(rows, question)
                if(len(true_rows) == 0 or len(false_rows) == 0):
                    continue
                gain = self.information_gain(true_rows, false_rows, uncertainity)
                
                if gain >= best_gain:
                    best_gain, best_question = gain, question
                    
        return best_gain, best_question
    def train(self,traindata):
        traindata = np.array(traindata)
        
        gain, question = self.split_data(traindata)
        if gain == 0:
            return Leaf(traindata)
        
        true_rows, false_rows = self.partition(traindata, question)
        
        true_branch = self.train(true_rows)
        false_branch = self.train(false_rows)
        
        return DecisionNode(question, true_branch, false_branch)
        
        
    def classify(self,row, node):
        """See the 'rules of recursion' above."""
    
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.predictions
    
        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)
    
    def predict(self,row, node):
        rows = np.array(row)
        probsvalue = []
        for row in rows:
            counts = self.classify(row,node)
            """A nicer way to print the predictions at a leaf."""
            total = sum(counts.values()) * 1.0
            
            probs = {}
            for lbl in counts.keys():
                probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
            probsvalue.append(probs)
        return probsvalue
        


class Question(DTreeClassifier):
    
    def __init__(self, column, value):
        self.column = column
        self.value = value
        
    def match(self,example):
        value = example[self.column]
        if self.is_numeric(value):
            return value >= self.value
        else:
            return value == self.value
   