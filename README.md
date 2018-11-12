# Machine Learning

## COMP

## Coursework1: Supervised Learning and Neural Networks

This coursework comprises 10% of the marks for this module.

Deadline: **21** / **11/2018** at 10:00 am, by electronic submission in Minerva.

Implement the following specification in Python. Submit a single zip file, containing exactly two
files, the Python code, and a pdf report.

The report must document your process and show the results, as specified in each section.

## Introductions

The Fashion-MNIST dataset is a dataset of Zalando articles’ images. It is composed of 70000
grayscale images of different fashion products, and their correspondent labels. For each category
of fashion item there are thousands of images.

## Prerequisite

In this Coursework you will design and train Artificial Neural Networks (ANNs) to perform image
recognition of fashion items.

In order to run the code necessary for this coursework, you need to install the following packages:

- numpy
- sklearn
- matplotlib
- tensorflow==1.
- keras

Once the installation is completed, run the provided Python script “CW_START.py”, which will
display the first 20 images and labels of the dataset you will use for this coursework.

The dataset is loaded as a _Training/Validation set_ (60 000 images and labels) and a _Test set_ (
000 image and labels). We will then further split the _Training/Validation set_ into a _Training set_ and a
_Validation set_.

## Design of an ANN with Keras

One of the most important tasks in the design of an Artificial Neural Network is finding a good
starting architecture. This is usually a trial-and-error process, as there exist no general procedure
for approaching this problem.

You are going to use Keras (https://keras.io/), one of the most popular libraries for the design of


Deep Artificial Neural Networks. You will find the description of the basic parts of this library, which
you will use for your exercises, in the next sections of this coursework. Nonetheless, feel free to
consult the official documentation at the link above.

The type of data from that we want our ANN to learn is usually the best starting point for this
design, together with the type of computation we would like to perform.

### Conv2D

The most popular solutions in image processing, for feature-extraction purposes, are the
Convolutional Neural Networks. These can be easily designed by using different instances of
Keras.layers.Conv2D which implements 2D convolutional layers. Here is a simplified example of
use:

Conv2D(filters=32, kernel_size=(3, 3), activation='relu',input_shape=(28,28,1), padding='same')

- Filters is the number of outputs we want to give to this specific layer.
- Kernel_size are the two integers specifying the dimension of the kernel matrix. This is also
    our matrix of trainable weights for this layer.
- Activation is the activation function we want to use in our artificial neurons and ‘relu’ is one
    of the most popular ones.
- Input_shape the shape of the input to this layer. It is usually a matrix whose last element is
    the channel dimension which in our case cane be set to 1. This should be set up only if the
    Conv2D layer is the input layer.
- Padding is the type of padding we want to apply to the output of our convolutional layer. It
    can be left to ‘same’.

### MaxPooling2D

Convolutional layers are usually followed by a downsampling step, which can be performed
through keras.layers.MaxPooling2D.

MaxPooling2D(pool_size=(2, 2), padding='same')

- Pool_size are two integers that specify the dimension of the downscaling.
- Padding is the type of padding we want to apply to the output of our convolutional layer. It
    can be left to ‘same’.

### Dropout

One of the main problems when using function approximators is balancing the neural network
architecture in such a way that it does not overfit the data. Usually this is guaranteed through the
usage of a Validation set, but with big architectures, often, this is not enough. One powerful and
broadly used technique is the _dropout_. This is simply about randomly _deactivating_ (setting to 0) a
certain fraction of the weights of the network at learning time. In Keras, this is possible using
keras.layers.Dropout.

Dropout(rate=0.01)

- Rate is the percentage of input values to this layer that will be set to zero at each learning
    step, that is, they will not contribute to the computation of the output of this layer.


The combination of these three Keras layers is able to perform a robust feature extraction
exploiting the power of Convolutional Neural Networks. Sometimes this is not enough and you
could want to pass through a further Convolutional unit.

Once the feature extraction is designed, we want a Multi-layer Perceptron (MLP) to elaborate the
features and perform the classification.

### Flattening

The first step to elaborate the extracted features is making them available to an MLP. The output of
the convolutions is an activation layer, which is a matrix. An MLP, however, processes one-
dimensional inputs. In order to give the activation layers as input to the MLP we need to stretch
them into a single (long) vector with: keras.layers.Flatten.

### Dense Layers

We are now ready to design our MLP for classification. In order to create one layer of such an
architecture, you can use keras.layers.Dense.

Dense(units=32, activation='relu')

- Units is the number of neurons in the layer.
- Activation is the activation function of the neurons, and ‘relu’ is one of the most popular
    ones (we saw another one that was easy to differentiate in class: the sigmoid).

As in the convolutional case, after a dense layer we might want to use a keras.layers.Dropout layer
in order to be robust against overfitting.

## Question 1

Start by opening the given **CW_START.py**. You will find an initial Neural Network architecture
developed in Keras following the instructions above.

Launch the training of the given ANN.

**1.a:** Add to the report the plot of the accuracy and loss over time that the code generates, and
comment on the result. Is the loss converging on training and validation? Is the network overfitting?

```
[4 marks]
```
**1.b:** Experiment with different learning rates and report on their effect on the loss over the training
and validation data. Identify three values of the learning rate, such that one is too high, one is too
law, and the third one is the one that you consider most appropriate. Include the corresponding
plots in the report, with a comment on the effect of the learning rate on the graphs.

```
[6 marks]
```
## Question 2

For the second problem you need to modify the current architecture. We propose some possible
modifications you should test yourself, in order to select the architecture that you consider best:

- In Conv2D layers: change the dimension of the filter.
- In Conv2D layers: change the number of filters.


- Change the dropout value in Dropout layers.
- In dense layers: change the number of neurons in the layer.
- Change the number of dense layers.
- Change the hyperparameters.

**2.a** : Design an architecture that achieves a loss on the _training_ data that is as far below 0.28 as
you can. Add the corresponding plots to the report, and describe the architecture, and the
reasoning behind your changes.

```
[8 marks]
```
**2.b** : Design an architecture (different from the one above) that achieves a loss on the _validation_
data that is as far below 0.27 as you can. Add the corresponding plots to the report, describe the
architecture, and the reasoning behind your changes.

```
[10 marks]
```
**2.c** : Use both architectures from the points above on the _test_ data, and comment on which one
performs best. Hint: the results on the test set are currently printed in the terminal at the end of
training.

```
[6 marks]
```
**2.d** : Build the confusion matrix of your best architecture over the test set, and add it to the report.
Which class is the easiest and which one the most difficult to classify?

```
[6 marks]
```

