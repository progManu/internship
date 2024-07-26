# Internship

Welcome to the Internship Experiments repository! This repository contains various experiments and projects conducted during the internship program.

## Table of Contents
- *Experiment 1*: ...
- *Experiment 2*: The idea behind the experiment 2 is to test how much the invariance forced by a global pooling affects the data separation ability of a neural network. In our case we created a toy dataset with 2 classes (A: low-high, B: high-low); the idea is that we want also to know if these classes are placed on the right or the left of array (imagine a picture with one row of pixels). What we would expect is that with different experimental setting (mainly adding and removing the **Global Pooling** operation) the performaces should change drastically.
    - **phase 1**: train a simple NN with one conv layer and one FC layer
    - **phase 2**: take the trained conv layer and apply a global pooling in order to get one value; dicretize the ranges and try to use it to classify
    - **phase 3**: take the trained conv layer use the global pooling in order to reduce the number of values after the convolution to 2 (basically two classes) but leave 4 outputs in the FC layer
    - **phase 4**: train another NN with the global pooling with 2 outputs and and 2 FC outputs and test the separablity (used K-means)
    - **phase 5**: train another NN with the global pooling with 1 output and and 2 FC outputs and test the separablity (used K-means)
    - **phase 6**: train another NN with the global pooling with 1 output and and 4 FC outputs and test the separablity (used K-means)

    *Result*: what I think is happening is that when the data erased from the global pooling is comparable to the intrinsic dimension of the data we have a huge loss of information as we can see from phase 5 and 6

- *Experiment 2.1*: This is a revision of the experiment 2 with just the phase 1 and 6 architectures and adding the possibility to use AvgPool
    - **phase 1**: train a simple NN with one conv layer and one FC layer
    - **phase 2**: train the head of above NN with global pooling with 1 output and and 4 FC outputs
    - **phase 3**: train the head of above NN with avg pooling with 1 output and and 4 FC outputs
    - **phase 4**: train another NN with the global pooling with 1 output and and 4 FC outputs
    - **phase 5**: train another NN with the avg pooling with 1 output and and 4 FC outputs

- *Experiment ModMNIST*: This experiment is an extension of the second one with a bigger dataset (ModMNIST is a version of MNIST where each image is scaled down to 14*14 and put in one of the 4 corners of the image eg. 1 put in the top left corner; the labels of this dataset are adjusted accordingly 4 position * 10 digist = 40 classes)
    - **phase 1**: train a simple NN with one conv layer and one FC layer
    - **phase 2**: train the head of above NN with avg pooling with 1 output for each channel
    - **phase 3**: train the head of above NN with histogram pooling with 1 output for each channel
    - **phase 4**: train (with different loops one for digits and one for postions) head of above NN with histogram pooling with 1 output for each channel
    - **phase 5** train the head of the above NN with avg pooling with 1 output for each channel but with more FCN layers
    - **phase 6**: repeat **phase 1** and **phase 2** with a bigger CNN

- [Contributing](https://github.com/progManu)
## Project structure

