---
layout: post
title:  "003 Experiment wih Simple Autoencoders"
date:   2017-02-03 19:46:31 +0100


categories: Experiment
---

## Introduction

Our problem stays the same, namely user authentication using machine learning. This problem was already idenitified as a classification problem and further reduced to an autentication problem. An authentication problem, where we need to decide wether a given canvasdrawing is from the given user. This has only two answers: Yes or No. 
From [the previous experiment](https://cmaixen.github.io/Masterthesis/experiment/2017/02/03/002_Experiment_with_PCA_and_NN_Classifier.html) we saw that the preprocessing by PCA did not improve the classification with a K-NN-classifier.

The next thing we will try are Autoencoders. Autoencoders are basically a special variant of Neural Networks (NN), where the input layers has the same amount of nodes as the output layers, with the purpose to reconstruct our own input. We believe after reading some literature that autoencoder will be better in reducing the features of a canvas than with PCA.  

The interessting part for us lies in the hidden layer, where the amount of nodes is the lowest and where the NN tries to  reconstruct the input from. This hidden layer is also called ***"the bottleneck layer"***. The following picture illustrates the setup of an Autoencoder. 

![Illustration of an autoencoder](http://nghiaho.com/wp-content/uploads/2012/12/autoencoder_network1.png)


## Experimental Setup
The goal of this experiment is to extract the feature representation by the autoencoder and use them to classify the canvases. 

The experiment is done by feeding the canvases to and fitting the autoencoder. After the fitting, we can retrieve for every canvas the feature representation by the bottleneck layer.
These extracted features will than be used for the classification of the canvases by the K-NN-Classifier.

### Autoencoder

For the implementation of the autoencoder we will use two libraries: [Tensorflow]() and [Keras](). 

* **Tensorflow** is an open-source software library for Machine Intelligence and a popular library in the research field.

* **Keras** is a high-level neural networks library, written in Python and capable of running on top of TensorFlow. It simplifies the coding.


During the experiment, we configure the autoencoder to do **100 epochs**, **shuffle** the data and apply a **batch_size of 256**. We log also everything with Tensorboard. This enables us to see the progress during the epochs.

In code: 

```python
 autoencoder.fit(X_train, X_train,
                        nb_epoch=100,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(X_test,X_test),
                        callbacks = [TensorBoard(log_dir='./results/003_Simple_Autoencoder')])
```



### Dataset


The dataset is the same as in previous experiments.

We have collected canvases from multiple device from multiple users. 

We will compose the dataset by picking 2000 canvases from a random user with one random device from the database. If the device has no 2000 canvases redo the picking. Once we have selected a device, we label all the canvases with "Yes". Next we pick 2000 random other devices and label it with "No".

The resulting dataset is a set with 4000 canvases, with 50% Yes-lables and 50% No-labels.

### Validation

We will use 10-fold validation to evaluate the performance of the classifier.

### Classifier

We will use the K-Nearest Neighbour classifier to classify the data, with the optimal configuration of [experiment 001](https://cmaixen.github.io/Masterthesis/experiment/2017/01/30/001_Experiment_with_NN_classifier.html), namely K=10

## Result

At the beginning we overestimated the power of the neural network and the available computational power of the cluster. We connected the inputlayer with 180k inputnodes with a Dense encoding layer. We soon discovered that this took way to much memory and we had to reduce the amount of nodes.

The naive implementation looked as the following:  

	```python
	input_img = Input(shape=(180000,))
	# "encoded" is the encoded representation of the input
	encoded = Dense(encoding_dim, activation='relu')(input_img)
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(180000, activation='sigmoid')(encoded)
	```

To reduce the amount of nodes, we applied Maxpooling with a 6x6 filter on the horizontal and vertical axis. The border_mode was set to "same", which means that you get as output the same size as the input. 

	```python
	input_img = Input(shape=(300,150,4))
	encoded = MaxPooling2D((6, 6), border_mode='same')(input_img)
	decoded = UpSampling2D((6, 6))(encoded)
	```
The following table gives the accuracy achieved by the classifier with the features of the autoencoder.


| # fold | Average Accuracy |
|--------|------------------|
| 1      | 0.4925           |
| 2      | 0.5475           |
| 3      | 0.485            |
| 4      | 0.4975           |
| 5      | 0.5475           |
| 6      | 0.545            |
| 7      | 0.51             |
| 8      | 0.5225           |
| 9      | 0.5              |
| 10     | 0.49             |


*	**Standaard deviation:** 0.0215
* 	**Average Accuracy:** 0.51375

These are really bad results. It performance less or equal as a random classifier.

If we look at our plots of the loss and value-loss for the different folds over the amount of steps (screenshots from tensorboard), we see for every fold a constant and no improvemnt over the amount of steps. 

![Loss](https://github.com/cmaixen/Masterthesis/blob/master/_images/003_simple_autoencoder_loss_function.png)

![Value Loss](https://github.com/cmaixen/Masterthesis/blob/master/_images/003_simple_autoencoder_valueloss_function.png)

After analysing the results and reviewing our experimental setup, there are several reason which could explain the bad results.

1.  First of all the downsampling of the samples. With downsampling we **must have spatial invariance**. This is not the case with our canvases. We know already from the previous experiment, the spatial variance is important.

2. Secondly we feed the whole set to the autoencoder but a better thing would be to feed it per user to the autoencoder and use a** per-user-encoding** to learn the authentication of a certain user.

3. A last argument is that we may want to change up the technique a little bit. By extracting every time the most dominant features we might lose the most important information. Because these dominant features may be for every canvas from every user the same. This gives us a sort of common feature set, which makes it very difficult to seperate different canvaseses from different users. We can compare this with "**Signal to Noise problem**", where we try to compare seperate the signal from the noise. Currently we are looking for the signal, but maybe the key to our problem is the noise. The noise could be the identifying part for the canvases of a certain user. We will do further research on this.

The following sketch illustrates the idea of the "**identifying noise**"

![Identifying noise](https://github.com/cmaixen/Masterthesis/blob/master/_images/003_simple_autoencoder_sketch.png)

## Conclusion

As a conclusion we can say that the current configuration of the autoencoder is not working. In the following experiments we will apply suggestion 2, were we will use a per-user-encoder. Also during the proces of reviewing the results, we got a new idea to have a better insight in the canvases. We could look at the problem on device level and use fully convolutional networks. Because we have labeled data, we could apply supervised learning and analyse how the NN performs on canvases from a device it already has encountered and on canvases from a new device. The **identifying noise** idea is something we keep in mind during future experiments.






