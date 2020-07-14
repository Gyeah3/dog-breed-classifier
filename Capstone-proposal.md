# Project Overview

This project is all about the classic Machine Learning classification. Given a dataset of random images, our model should be able to detect the breed of the dog if a dog is present in the image. We are going to use CNN (Convolutional Neural Network) to build this model.

Initially we are going to import some of the necessary modules required for this project. Then we are going to download the required datasets containing images required for this project. Firstly we are going to try and detect human and dog faces in the datasets before proceeding to the project to get an initial idea. The data is then pre-processed by horizontally flipping, cropping the images and so on and so forth. Python Libraries like TensorFlow and openCV to detect the faces in the images.

We are going to use CNN to classify dog breeds (using Transfer Learning), then write, train and test the algorithm.

## Problem Statement

When a random image is fed into the model, it will classify the image to detect the dog and it's breed in the image and return to the user.

## Importing Modules and datasets

Link to the datasets https://github.com/Gyeah3/dog-breed-classifier-udacity/tree/master/project-dog-classification/images

Pytorch will be the main machine learning (ML) library used for this project. The benchmark model for this project will be the VGG-16, a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford, this model achieved 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. Achieving 92.7% accuracy for this project with the small dataset available (8351 images of dogs) in comparison to the image net data set which the VGG-16 was trained is definitely a far reach, but it’s something to aim for. First we are going to import all the necessary modules and libraries required for the project.

We are going to detect the human faces and dog faces in the image datasets by converting the images into numpy arrays.

## Solution

We are going to create our own **Convolutional Neural Network (CNN)** to tackle this problem of detecting the dogs in the images and classifying the breed of the dog. We will train the model using the CNN (Transfer Learning), particularly **ResNet** to train the model for an improved accuracy.

## Benchmark

The Benchmark for this project is going to be the pre-trained **VGG-16 model** which is a very popular image dectector trained on a very large dataset.
