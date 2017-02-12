---
layout: post
title:  "001 Experiment with NN Classifier"
date:   2017-01-30 19:46:31 +0100


categories: Experiment
---

# Experimental Setup

## Problem

The classifcation problem is reduced to the authentication problem, where we need to decide wether the given canvas is from the given user. This has only two answers: Yes or No. 

## Dataset

We have collected canvases from multiple device from multiple users. We know that small changes in the browser, for example the change in resolution, can allow the user to submit the device twice. For this reason we will only consider one device of each user for in the dataset. 

We will compose the dataset by picking 2000 canvases from a random user with one random device from the database. If the device has no 2000 canvases redo the picking. (Optionally we can also reduce the amount of canvases and allow an incomplete set of canvases. a set is considered incomplete if it has less than 2000 canvases). Once we have selected a device, we label all the canvases with "Yes". Next we pick 2000 random other devices and label it with "No".

The resulting dataset is a set with 4000 canvases, with 50% Yes-lables and 50% No-labels.

## Validation

We will use 10-fold validation to evaluate the performance of the classifier.



```python

kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(x_data):
    print("fold: " + str(counter))
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    
    for i in train_index:
        X_train.append(x_data[i])
        Y_train.append(y_data[i])
        
    for i in test_index:
        X_test.append(x_data[i])
        Y_test.append(y_data[i])
            
```
        

## Classifier

We will use the Nearest Neighbour classifier to classify the data


```python
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(X_train,Y_train)
Y_predict = clf.predict(X_test)
            
```





