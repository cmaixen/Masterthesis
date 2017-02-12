---
layout: post
title:  "001 Experiment with NN Classifier"
date:   2017-01-30 19:46:31 +0100


categories: Experiment
---

## Experimental Setup

### Problem

The classifcation problem is reduced to the authentication problem, where we need to decide wether the given canvas is from the given user. This has only two answers: Yes or No. 

### Dataset

We have collected canvases from multiple device from multiple users. We know that small changes in the browser, for example the change in resolution, can allow the user to submit the device twice. For this reason we will only consider one device of each user for in the dataset. 

We will compose the dataset by picking 2000 canvases from a random user with one random device from the database. If the device has no 2000 canvases redo the picking. (Optionally we can also reduce the amount of canvases and allow an incomplete set of canvases. a set is considered incomplete if it has less than 2000 canvases). Once we have selected a device, we label all the canvases with "Yes". Next we pick 2000 random other devices and label it with "No".

The resulting dataset is a set with 4000 canvases, with 50% Yes-lables and 50% No-labels.

The following small report, gives you an idea about the data in the database.

```text
CANVAS DATABASE STATISTICS
--------------------------
                
General Stats
------------
# Devices: 80
# Users: 49
# Canvases: 147387
Completed Devices: 87.5 %
  
OS STATS
----------
Windows 8.1 : 2.5 %
Mac OS X : 26.25 %
Linux : 13.75 %
iOS : 17.5 %
Ubuntu : 8.75 %
Windows 10 : 8.75 %
Windows 7 : 3.75 %
Android : 18.75 %
 
Browser STATS
--------------
Firefox : 11.25 %
Chrome : 32.5 %
IE : 1.25 %
Mobile Safari : 17.5 %
Chrome Mobile : 13.75 %
Samsung Internet : 1.25 %
Firefox Mobile : 3.75 %
Edge : 1.25 %
Safari : 13.75 %
Chromium : 2.5 %
Iceweasel : 1.25 %
 
Mobile STATS 
--------------
is mobile - No :66.25 %
is mobile - Yes :33.75 %
 
Navigator Platform
------------------
iPad : 2.5 %
Linux x86_64 : 22.5 %
MacIntel : 26.25 %
Win32 : 15.0 %
Linux armv7l : 11.25 %
iPhone : 15.0 %
Linux armv8l : 7.5 %
 
          
Report generated on : 2017-01-24 11:47:23
```


### Validation

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
        

### Classifier

We will use the Nearest Neighbour classifier to classify the data

```python
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(X_train,Y_train)
Y_predict = clf.predict(X_test)           
```


## Results

The Nearest Neighbour classifier has different options for the weight: 
	* ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
	* ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.

First we try to optimise the configuration for K and the weights.

| n_neighbours | Average Accuracy | Std             | Weights  |
|--------------|------------------|-----------------|----------|
| 2            | 0.556            | 0.0222822799552 | uniform  |
| 2            | 0.59375          | 0.0267920230666 | distance |
| 4            | 0.5825           | 0.0204633819297 | uniform  |
| 4            | 0.6125           | 0.0170660481659 | distance |
| 5            | 0.6185           | 0.0133790881603 | uniform  |
| 5            | 0.62175          | 0.0272729261356 | distance |
| 10           | 0.58675          | 0.0219673052512 | uniform  |
| 10           | 0.625            | 0.0249499498997 | distance |
| 12           | 0.57975          | 0.0173727516531 | uniform  |
| 12           | 0.6145           | 0.0303685692781 | distance |
| 15           | 0.6025           | 0.0141862609591 | uniform  |
| 15           | 0.609            | 0.0219146070008 | distance |
| 30           | 0.58075          | 0.0291129266821 | uniform  |
| 30           | 0.609            | 0.0235637433359 | distance |



![001A_Experiment_with_NN_classifier_all](https://raw.githubusercontent.com/cmaixen/Masterthesis/master/_images/001A_Experiment_with_NN_classifier_all.png)

Overall dominates 'distance' the configuration. For this configuration seems K=10 the optimal configuration. 

If we repeat the experiment with the the optimal configuration, namely weights="distance" and K=10, on 15 random composed datasets, we get the following results:

| Run | Average Accuracy | Std              |
|-----|------------------|------------------|
| 1   | 0.72825          | 0.0212735163995  |
| 2   | 0.54175          | 0.026670442441   |
| 3   | 0.74375          | 0.0208940781084  |
| 4   | 0.650337249782   | 0.031165912499   |
| 5   | 0.73525          | 0.0199515037027  |
| 6   | 0.5385           | 0.0266505159425  |
| 7   | 0.73825          | 0.0221091044595  |
| 8   | 0.641            | 0.0219715725427  |
| 9   | 0.71125          | 0.0120026038842  |
| 10  | 0.65375          | 0.0170018381359  |
| 11  | 0.58325          | 0.0230502169187  |
| 12  | 0.6495           | 0.0219317121995  |
| 13  | 0.956475279107   | 0.00842988336461 |
| 14  | 0.6485           | 0.027527259217   |
| 15  | 0.80775          | 0.0189225394702  |

![001B_Experiment_with_NN_classifier_barchart](https://github.com/cmaixen/Masterthesis/blob/master/_images/001B_Experiment_with_NN_classifier_barchart.png?raw=true)

![001B_Experiment_with_NN_classifier_boxplot](https://github.com/cmaixen/Masterthesis/blob/master/_images/001B_Experiment_with_NN_classifier_boxplot.png?raw=true)



Overall does the random classifier perfom better than random with an average of 69%. But we see that the data is widly spread.

## Conclusion 

As a conclusion, we can say that the Nearest Neighbor classifier performs better than random and is already a good start towards a solution for the research. Althought we must remember that our database is not a one-on-one reflection of the real world and does not represent all devices in the real world. Also the variation in performance for certain dataset is something to notice. We want a consistent accuracy. Also the representation of of a cansvas can be improved. Currently a canvas is represented by flattened matrix of the RGBA-values, which results in 180k features (size of a canvas is 300 x 150). 180k feautres is to much and has to be reduced. In [the next experiment](https://cmaixen.github.io/Masterthesis/experiment/2017/02/03/002_Experiment_with_PCA_and_NN_Classifier.html) we will try to this. 

Overall we can say that The Nearest Neighbours Classifier is a good start, but there is much room for improvement.







