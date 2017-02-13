---
layout: post
title:  "002 Experiment with PCA and NN Classifier"
date:   2017-02-03 19:46:31 +0100


categories: Experiment
---

## Experimental Setup

The classifcation problem is reduced to the authentication problem, where we need to decide wether the given canvas is from the given user. This has only two answers: Yes or No. 
With [the previous experiment](https://cmaixen.github.io/Masterthesis/experiment/2017/01/30/001_Experiment_with_NN_classifier.html) we saw that just the RGBA-values may not be informative enough. This is why we extract first the most informative features with PCA and use these with the classifier.

### Dataset

We have collected canvases from multiple device from multiple users. We know that small changes in the browser, for example the change in resolution, can allow the user to submit the device twice. For this reason we will only consider one device of each user for in the dataset. 

We will compose the dataset by picking 2000 canvases from a random user with one random device from the database. If the device has no 2000 canvases redo the picking. (Optionally we can also reduce the amount of canvases and allow an incomplete set of canvases. a set is considered incomplete if it has less than 2000 canvases). Once we have selected a device, we label all the canvases with "Yes". Next we pick 2000 random other devices and label it with "No".

The resulting dataset is a set with 4000 canvases, with 50% Yes-lables and 50% No-labels.

### Validation

We will use 10-fold validation to evaluate the performance of the classifier.

### Classifier

We will use the Nearest Neighbour classifier to classify the data, with the optimal configuration of [the previous experiment](https://cmaixen.github.io/Masterthesis/experiment/2017/01/30/001_Experiment_with_NN_classifier.html), namely K=10

As a train- and testset will be pre-processed by the PCA before we feed it to the classifier

```python
pca = PCA(n_components=j)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```

## Result

The first results display the classification accuracy in function of the amount of features. 

| n_components | Average Accuracy | Std             |
|--------------|------------------|-----------------|
| 2            | 0.52925          | 0.0283736233146 |
| 4            | 0.53225          | 0.0204465767306 |
| 8            | 0.55225          | 0.0261593673471 |
| 16           | 0.58825          | 0.02205249419   |
| 32           | 0.599            | 0.028464890655  |
| 64           | 0.59175          | 0.0160876505432 |
| 128          | 0.5985           | 0.016590660023  |
| 256          | 0.60775          | 0.025407921993  |
| 512          | 0.594            | 0.0263912864408 |
| 1024         | 0.59             | 0.0219943174479 |
| 2048         | 0.629            | 0.0297321374946 |
| 4096         | 0.6525           | 0.0163935963108 |
| 8192         | 0.6495           | 0.0170953209973 |
| 16384        | 0.64825          | 0.0172137299851 |
| 32768        | 0.65275          | 0.0161031829152 |
| 65536        | 0.6545           | 0.0159608896995 |
| 131072       | 0.64975          | 0.0203853501319 |





![001B_Experiment_with_NN_classifier_boxplot](https://github.com/cmaixen/Masterthesis/blob/master/_images/002A_Experiment_with_PCA_and_NN_Classifier.png?raw=true)

In the results we see that there is a stagnation after an amount of 4096 features. Increasing the amounts doesn't contribute to the accuracy.

If we repeat the experiment on 15 random generated datasets with the optimal configuration with the least optimal solution. We get the following results:


| Run | Average Accuracy | Std              |
|-----|------------------|------------------|
| 0   | 0.6645           | 0.0285438259524  |
| 1   | 0.6655           | 0.013955285737   |
| 2   | 0.65725          | 0.0219786828541  |
| 3   | 0.826            | 0.0192093727123  |
| 4   | 0.5345           | 0.0328024389337  |
| 5   | 0.629            | 0.0168151717208  |
| 6   | 0.995024875622   | 0.00385371477235 |
| 7   | 0.956479835954   | 0.0182139117714  |
| 8   | 0.7355           | 0.0114455231423  |
| 9   | 0.644759701633   | 0.0217405314397  |
| 10  | 0.5785           | 0.0211896201004  |
| 11  | 0.53425          | 0.0253734605444  |
| 12  | 0.6235           | 0.0235902522242  |
| 13  | 0.7205           | 0.0230976189249  |
| 14  | 0.66575          | 0.0191066088043  |

![001B_Experiment_with_NN_classifier_boxplot](https://github.com/cmaixen/Masterthesis/blob/master/_images/002A_Experiment_with_PCA_and_NN_Classifier.png?raw=true)

![001B_Experiment_with_NN_classifier_boxplot](https://github.com/cmaixen/Masterthesis/blob/master/_images/002A_Experiment_with_PCA_and_NN_Classifier.png?raw=true)

First of all is it interesting to see that we have for certain datasets an extremly high precision. It is interessting to take a closer look to that in further research.
Our average acuracy has slightly increased with 1% compared to the results without PCA. Not the big improvement we hoped for. Also do we see there are still 


## Conclusion
