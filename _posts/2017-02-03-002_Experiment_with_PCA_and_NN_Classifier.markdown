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

![001B_Experiment_with_NN_classifier_boxplot](https://github.com/cmaixen/Masterthesis/blob/master/_images/002C_Experiment_with_PCA_and_NN_Classifier_barchart.png?raw=true)

#### BOXPLOT NN Classifier WITH PCA
![001B_Experiment_with_NN_classifier_boxplot](https://github.com/cmaixen/Masterthesis/blob/master/_images/002C_Experiment_with_PCA_and_NN_Classifier_boxplot.png?raw=true)

#### BOXPLOT NN Classifier WITHOUT PCA (from [experiment 001]((https://cmaixen.github.io/Masterthesis/experiment/2017/01/30/001_Experiment_with_NN_classifier.html))
![001B_Experiment_with_NN_classifier_boxplot](https://github.com/cmaixen/Masterthesis/blob/master/_images/001B_Experiment_with_NN_classifier_boxplot.png?raw=true)


First of all is it interesting to see that we have for certain datasets an extremly high precision. It is interessting to take a closer look to that in further research.
Our average acuracy has slightly increased with 1% compared to the results without PCA. Not the big improvement we hoped for. Also if we compare the boxplots of this experiment and the previous experiment, we see that the interquartile for "With PCA"  is bigger. Meaning that the consistence accuracy over the dataset is less than "Without PCA". 

## Conclusion

PCA results in a very small improvement sand shows not the improvements we hoped for. At last do we see also a wider spreading of the accuracy. A consistence accuracy is also something we like to work to and PCA does not fulfill this requirement.

##UPDATE(25/02): BUG Found 

An explanation is found for the extreme outliers at run #6 and #7 and possible #9 in the experiment with 15 random generated datasets.

We found a bug in the generation of the dataset. In the selectionproces of the device to label with yes went sometimes wrong because of a wrong comparison.

```python
      if len(selected_canvases) <= 2000:
            print(len(selected_canvases))
            full_set = True

```

This resulted in a unbalanced train/testset. Sometimes we had the selection of an incomplete set (set with less than 2000 samples) and a incomplete amount of 'Yes'-samples. By checking the logs of the experiment our finding gets confirmed.

| Run | Size Yes-labels | Size No-labels | Size X_data | Size Y_data |
|-----|-----------------|----------------|-------------|-------------|
| 0   | 2000            | 2000           | 4000        | 4000        |
| 1   | 2000            | 2000           | 4000        | 4000        |
| 2   | 2000            | 2000           | 4000        | 4000        |
| 3   | 2000            | 2000           | 4000        | 4000        |
| 4   | 2000            | 2000           | 4000        | 4000        |
| 5   | 2000            | 2000           | 4000        | 4000        |
| **6**   | **10**              | **2000**           | **2010**        | **2010**        |
| **7**   | **91**              | **2000**           | **2091**        | **2091**        |
| 8   | 2000            | 2000           | 4000        | 4000        |
| **9**   | **1561**            | **2000**           | **3561**        | **3561**        |
| 10  | 2000            | 2000           | 4000        | 4000        |
| 11  | 2000            | 2000           | 4000        | 4000        |
| 12  | 2000            | 2000           | 4000        | 4000        |
| 13  | 2000            | 2000           | 4000        | 4000        |
| 14  | 2000            | 2000           | 4000        | 4000        |



We need to check the amount of devices with exactly 2000 canvas samples to know the influence on the research.

If we check our database, we see that the algorithm with the bug had still 70 devices to choose from. Also we see that there are no devices which  have more than 2000 canvas samples.


| # Samples | # Devices |
|-----------|-----------|
| < 2000    | 10        |
| 2000      | 70        |
| > 2000    | 0         |
 
 
As a conclusion about the impact we can say that the bug did not exclude devices, but only included to much devices. The new results can be calculated by removing the results with an unbalanced dataset. We do not need to redo the experiment to get valid results.

## New Results

By removing the incorrect results, we have still 13 runs left.

![001B_Experiment_with_NN_classifier_boxplot](https://github.com/cmaixen/Masterthesis/blob/master/_images/002C_Experiment_with_PCA_and_NN_Classifier_boxplot_bug_fixed.png?raw=true)

As expected is the average lower and is our conclusion still valid, we cannot achieve a consistant accuracy with PCA and we need to search for other solutions.
 
 
