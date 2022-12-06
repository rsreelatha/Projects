#!/usr/bin/env python
# coding: utf-8

# # AIML Capstone Project Computer Vision - Car Detection

# Authors:
#    Ankit Garg     "ankitcse07@gmail.com"
#    Sandip Maity   "sandipmaity2006@gmail.com"
#    R. Sreelatha   "radha_sreelatha@yahoo.com"
#    Prateek Gupta  "prateek.jaypee@gmail.com"
#    N. Sugathri    "nsugathri@yahoo.com"

# Domain: Automotive Surveillance

# Problem Statement: Computer vision can be used to automate supervision and generate action appropriate action trigger if the event is predicted from the image of interest. For example a car moving on the road can be easily identified by a camera as make of the car, type, colour, number plates etc.

# Objective: The object of this project is to design a deep learning based model for car detection.

# Approach
# - Step 1: Import the data
# - Step 2: Map training and testing images to its classes.
# - Step 3: Map training and testing images to its annotations. 
# - Step 4: Perform Exploratory Data analysis
# - Step 4: Load images and preprocess  
# - Step 5: Design, train and test CNN models to classify the car and find the bounding boxes. This will also involve creating a pipeline involving data augmentation (like flip, rotate etc),  normalizing the image.  This would all be done online as dataset is large and will not fit in the memory. 
# - Step 6: Design, train and test RCNN & its hybrids based object detection models to impose the bounding box or mask over the area of interest
# - Step 7: Pickle the best model and use it for further prediction
# - Step 8: Design a clickable UI that can automate the tasks of importing the data, mapping the train/test images to its clases, train and build a modell and use the pickled model to predict based on an image input by the user
# 

# Data Description:
# The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. Tesla Model S 2012 or BMW M3 coupe 2012.
# Data description:
# ‣ Train Images: Consists of real images of cars as per the make and year of the car.
# ‣ Test Images: Consists of real images of cars as per the make and year of the car.
# ‣ Train Annotation: Consists of bounding box region for training images.
# ‣ Test Annotation: Consists of bounding box region for testing images.

# Evaluation:For evaluating the performance of the models, one or more measures as relevant will be captured.
# For classification:
# - Categorical Accuracy: Since the class labels will be one-hot encoded, Categorical accuracy will be calculated to measure how often predictions matches one-hot labels.
# 
# For the predictions involving bounding boxes:
# - Intersection over union (IoU)- a measure of overlap between the ground truth and predicted bounding boxes. A threshold value of IOU above 0.5 would be set. If the IOU is above the threshold the prediction is considered good.
# 
# - Precision and recall- While precision indicates what percentage of positive predictions are correct , Recall would indicate what percentage of ground truth objects were found.  True positives are calculated based on  Number_of_detection with IoU > threshold. False positives are number of detections with IoU<=threshold or detected more than once. Fasle Negative = number of objects that not detected or detected with IoU<=threshold.
# Precision = True positive / (True positive + False positive); Recall = True positive / (True positive + False negative)
# 
# - Average Precision: AP will be computed only for an object class on all images as AVG(Precision for each of 11 Recalls {precision = 0, 0.1, ..., 1}) 
# 
# - Mean average precision (mAP): Mean of the average Precision computed for all the classess on all the images. 
# 

# # Basic Configuration Information

# Conventions followed:
#     -project_path : identifies the path to the main project folder 
#     -data_path: Path where data stored with subfolders as defined below
#     -train_path: path where train data stored
#     -test_path: path where test data stored

# # The  project folder structure

# Under the project path save the data files as per the project structure
# 
# 
# Main Project folder (identified by project_path)
# 
# project_path
#     │
#     ├──data
#     │  │     anno_train.csv
#     │  │     anno_test.csv
#     │  │     names.csv
#     │  │
#        └── car_data
#            └── car_data
#                 └── train (with subfolders for the classes and corresponding images)
#                 └── test   (with subfolders for the classes and corresponding images)
# 

# # Import Basic Libraries
# Import the libraries and update the requirements.txt
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

## Import random and initialize the seed for deterministic results
import random
random.seed(42)

## Import visualizations package
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#get_ipython().run_line_magic('matplotlib', 'inline')


import seaborn as sns

from zipfile import ZipFile
import cv2
import os, sys

import pickle

## DL specific packages
import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

## Deep Learning Networks
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet121

## Import Different Layers
from tensorflow.keras.layers import  Flatten, Reshape, Dropout, UpSampling2D, Concatenate, Conv2D, Dense, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D

## Import GUI packages
#import gradio as gr
import streamlit as st


## Install the albumentations package 
## Please restart the notebook once more as this will not load
#pip install --user albumentations
import albumentations as A

from PIL import Image
import time


## Applicable if using googlecolab
#Mount google Drive 
#setting default value as False change it to True if using google colab
use_googl_colab = False
if(use_googl_colab):
    from google.colab import drive
    drive.mount('/content/drive')
    #specify the project path
    project_path = "/content/drive/My Drive/Colab Notebooks/AIML Capstone/"

    # change to the project folder
    os.chdir(project_path)

    # Add the path to the sys.path for the session
    sys.path.append(project_path)


# #### ATTENTION- Desktop users- check this section and update the main project path 

# set the project folder directory
project_path="../"
# #### Common directory structure 
#Setting Path to Data
#the following folderndata_path contains anno_train.csv, 
## anno_test.csv and names.csv files
data_path= project_path

# train_path contains the class specific images from the training data set
train_path=data_path+"/car_data/car_data/train/"
#test path contains the class specific images from the test data set
test_path=data_path+"/car_data/car_data/test"


# ## Input to the requirements.txt file  <to be updated if any new requirements identified>
#     
# ###### Basic packages- identify the versions and add
# * python=3.*
# * numpy=1.19
# * sklearn=0.24
# * Flask==1.1.2
# * pandas==1.2.1
# * tensorflow=2.4
# * albumentations=0.5.2

# # Load the train and test data and map the data to its classes

# * Step 1: Import the data from anno_train.csv, anno_test.csv files
# * Step 2: Map training and testing images to its classes. 
# * Step 3: Map training and testing images to its annotations. 
# * Step 4: Display images with bounding box.
# * Step 5: Import the data from names.csv file
# * Step 6: Perform exploratory Data analysis
# 
# In steps 2/3, we will also perform sanitization checks on the data like bounding boxes are smaller than the image size, folder names are same as that provided in the annotation file, also labels provided are matching as that of folder name. 
# 
# Since the train and test annovation .csv files does not have the headers,  while loading the data specifying the headings as identified in the stanford car data set. The fields are respectively
# ImageFileName: Name of the file containing the image of the car
# x1: Min x-value of the bounding box, in pixels
# x2: Max x-value of the bounding box, in pixels
# y1: Min y-value of the bounding box, in pixels
# y2: Max y-value of the bounding box, in pixels
# label_index: Value of the class label
# 
# **Bounding boxes given in the dataset are of pascal format, we will convert to coco format**. 
# 
# Since the names.csv file contains only the class name and does not have a header, header labels to be specified while loading the data and a label indexing matching with the class to be included.


### Let's declare all globals
global annot_train_df
global annot_test_df
global label_df
global metadata_train
global metadata_test
global ss



## Define bools for different tasks in GUI each should saved in the session state and restored back
EDATaskDone = False
MetaDataLoaded = False

def clearGUIAppTaskBools():
    global EDATaskDone
    global MetaDataLoaded
    
    EDATaskDone = False
    MetaDataLoaded = False

## This is only required to run on GPUs. GPUs can give lot of acceleration. In case GPU device is available, you will 
## find the gpu name here 
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

## Stream Little Set Title
st.title("AIML Capstone: Car Object Detection")

import PIL
## Some utility function to load images
def load_image(path):
    
    ## There is some problem with cv2, I found that cv2 is giving incorrect orientation
    ## reversing heightxwidth and hence bounding box become invalid, so switch to 
    ## PIL 
    #img = cv2.imread(path, 1)
    
    ## We need to convert to RGB format
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = PIL.Image.open(path).convert('RGB')
    return np.asarray(img)

## This function would be implemented when integrating with GUI

global progressBar
progressBarValue = 0
def updateProgressBar(value):
    """Keeping the place holder for updating the 
    Need to write custom code wrt api exposed
    by a particular gui platform."""
    
    global progressBarValue
    
    if(progressBarValue < 100):
        progressBarValue = progressBarValue + value
        progressBar.progress(progressBarValue)

## Let's write some helper function to load the annotation files
def loadAnnotDataFrame(filename):
    annot_df = pd.read_csv(filename, header=None)
    annot_df.columns = ['ImageFileName', 'x1', 'y1', 'x2', 'y2', 'label_index']
    annot_df.set_index('ImageFileName', inplace=True)
    return annot_df


##Load the respective frames for the bounding boxes, although we 
## can generate all information based on folder names, only thing that
## is missing is the bounding box information which is present in the csv provided
## with the data-set
def loadTrainTestAnnotFrames():
    global annot_train_df
    global annot_test_df
    global label_df
    annot_train_df = loadAnnotDataFrame(data_path + 'anno_train.csv')
    annot_test_df = loadAnnotDataFrame(data_path + 'anno_test.csv')
    
    label_df = pd.read_csv(data_path + 'names.csv', names=['carname'])
    #Since the index in csv file starts from 1 change the index in names file accordingly 
    label_df.index += 1
    #Changing the index name to label_index similar to the train and test data labels
    label_df.index.names = ['label_index']

    

from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import streamlit as st

## This code can capture the stdout/stderr in streamlit stream, which can 
## used to re-direct the logs
@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield


# # Exploratory Data Analysis¶

# As part of exploratory analysis will do :
# -a sanity check and ensure all the data has been imported correctly and the columns labelled appropriately
# -identify null values and duplicate values if any
# -identify the data types and if any further conversion is required
# -Identify the distribution of classes, whether any of the classes are over represented/under represented indicating imbalance in the class distribution that would affect the performance. 
# - Identify if data split equally among the train and test sets
# EDA can provide useful insights of the data and identify what further processing is required that can help build a robust model.
import io
def PerformEDA():
    global EDATaskDone
    loadTrainTestAnnotFrames()
    #Sanity check on the records, columns and the min/max labels, null values, data types in train data
    st.write("--------------PERFORM EXPLORATORY DATA ANALYSIS-------------------")
    
    st.write(f"------Summary of  train data set--------")
    
    strOutput = f"Image  annotations in the data set : {len(annot_train_df)}\n"
    strOutput += f"No of Columns in the data set : {len(annot_train_df.columns)}\n"
    strOutput += f"Total Number of labels = {annot_train_df.label_index.nunique()}\n"
    strOutput += f"Min Label Number = {annot_train_df.label_index.min()}\n"
    strOutput += f"Max Label Number = {annot_train_df.label_index.max()}\n"
    buffer = io.StringIO()
    annot_train_df.info(buf=buffer)
    strOutput += buffer.getvalue()
    strOutput += "\n\n"
    st.text(strOutput)
    
    
    #Sanity check on the records, columns and the min/max labels, null values, data types in test data
    st.write(f"-------Summary of test data set--------")      
    strOutput = f"Image annotations : {len(annot_test_df)}\n"
    strOutput += f"No of columns : {len(annot_test_df.columns)}\n"
    ## To check total labels,  min/max label number
    strOutput += f"Total Number of Unique labels= {annot_test_df.label_index.nunique()}\n"
    strOutput += f"Minimum Label Number = {annot_test_df.label_index.min()}\n"
    strOutput += f"Maximum  Label Number = {annot_test_df.label_index.max()}\n"
    buffer = io.StringIO()
    annot_test_df.info(buf=buffer)
    strOutput += buffer.getvalue()
    strOutput += "\n\n"
    
    
    duplicates_in_train=annot_train_df.index.duplicated()
    strOutput += f"Count of Duplicate rows in train data:{sum(duplicates_in_train)}\n"
    duplicates_in_test=annot_test_df.index.duplicated()
    strOutput += f"Count of Duplicate rows in test data:{sum(duplicates_in_test)}\n\n"
    st.text(strOutput)

    
    #Sanity check on the records, columns and the min/max labels, null values, data types in Names data
    st.write(f"-------Summary of Car Labels data set--------")     
    strOutput = f"Car labels available           : {len(label_df)}\n"
    strOutput +=  f"No of columns                 : {len(label_df.columns)}\n"
    ## To check total labels,  min/max label number
    strOutput += f"Total Number of unique labels  : {label_df.index.nunique()}\n"
    strOutput += f"Min Label Number               : {label_df.index.min()}\n"
    strOutput += f"Max Label Number               : {label_df.index.max()}\n"
    
    buffer = io.StringIO()
    label_df.info(buf=buffer)
    strOutput += buffer.getvalue()
    strOutput += "\n\n"

    ## Total number of duplicates in label names and total number of labels
    strOutput += f"Total duplicate Labels         : {label_df['carname'].duplicated().sum()}\n"
    strOutput += f"Total Number of labels         : {label_df['carname'].nunique()}\n"
    
    
    ## Let's find distribution of label in train/test dataset
    label_train_dist = annot_train_df.label_index.value_counts().sort_values(ascending=False)

    max_count_index = label_train_dist.index[0]
    strOutput += f"Train => Label = {max_count_index} has a maximum count of = {label_train_dist[max_count_index]}\n"
    min_count_index = label_train_dist.index[-1]
    strOutput += f"Train => Label = {min_count_index} has a minimum count of = {label_train_dist[min_count_index]}\n"
    
    st.text(strOutput)
    
    fig1 = plt.figure(figsize=(10.0,10.0))
    ax = fig1.add_subplot(221)
    sns.distplot(annot_train_df["label_index"],color="b",rug=True, ax=ax)
    ax.axvline(annot_train_df["label_index"].mean(),
            linestyle="dashed",color="g",
            label='mean',linewidth=2)
    ax.axvline(annot_train_df["label_index"].median(),
            linestyle="dashed",color="r",
            label='median',linewidth=2)
    Q1=annot_train_df.label_index.quantile(0.25)
    Q2=annot_train_df.label_index.quantile(0.75)

    plt.legend(loc="best",prop={"size":14})
    ax.set_title("Distribution of classes in train data")
    #plt.close()
    
    ax = fig1.add_subplot(222)
    sns.distplot(annot_test_df["label_index"],color="b",rug=True, ax=ax)
    ax.axvline(annot_test_df["label_index"].mean(),
            linestyle="dashed",color="g",
            label='mean',linewidth=2)
    ax.axvline(annot_test_df["label_index"].median(),
            linestyle="dashed",color="r",
            label='median',linewidth=2)
    Q1=annot_test_df.label_index.quantile(0.25)
    Q2=annot_test_df.label_index.quantile(0.75)

    plt.legend(loc="best",prop={"size":14})
    ax.set_title("Distribution of classes in test data")
    
    ax = fig1.add_subplot(223)
    annot_train_df.label_index.plot.hist(ax=ax)
    ax.set_title("Distribution of classes across train data")
    ax.set_xlabel('Class Labels')
    ax.set_ylabel('Frequency of classes');
    
    ax = fig1.add_subplot(224)
    annot_test_df.label_index.plot.hist(ax=ax)
    ax.set_title("Distribution of classes across test data")
    ax.set_xlabel('Class Labels')
    ax.set_ylabel('Frequency of classes');

    
    plt.close()
    st.pyplot(fig1)
    
    fig1 = plt.figure(figsize=(2.0,2.0))
    ax = fig1.add_subplot(111)
    ax.hist([annot_train_df.label_index, annot_test_df.label_index],label=['train', 'test'])
    plt.legend(loc='upper right')
    ax.set_title("Distribution of classes across test data")
    ax.set_xlabel('Class Labels')
    ax.set_ylabel('Frequency of classes')

    plt.close()
    st.pyplot(fig1)
    EDATaskDone = True

import SessionState


dataset_mean = [0,0,0]
dataset_std = [0,0,0]
total_pixels = 0 
def populateMeanForTheDataset(path):
    global total_pixels
    global dataset_mean
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            img_path = os.path.join(path, i, f)
            img = load_image(img_path)
            dataset_mean[0] = np.sum(img[:,:,0]) + dataset_mean[0]
            dataset_mean[1] = np.sum(img[:,:,1]) + dataset_mean[1]
            dataset_mean[2] = np.sum(img[:,:,2]) + dataset_mean[2]
            total_pixels = img.shape[0]*img.shape[1] + total_pixels
    dataset_mean[0] = (dataset_mean[0]/total_pixels)
    dataset_mean[1] = (dataset_mean[1]/total_pixels) 
    dataset_mean[2] = (dataset_mean[2]/total_pixels)


def populateStdForDataset(path):
    global total_pixels
    global dataset_mean
    global dataset_std
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            img_path = os.path.join(path, i, f)
            img = load_image(img_path)
            
            dataset_std[0] = np.sum((img[:,:,0] - dataset_mean[0])**2) + dataset_std[0]
            dataset_std[1] = np.sum((img[:,:,1] - dataset_mean[1])**2) + dataset_std[1]
            dataset_std[2] = np.sum((img[:,:,2] - dataset_mean[2])**2) + dataset_std[2]

    dataset_std[0] = np.sqrt(dataset_std[0]/total_pixels)
    dataset_std[1] = np.sqrt(dataset_std[1]/total_pixels) 
    dataset_std[2] = np.sqrt(dataset_std[2]/total_pixels)

## Both mean/std needs to be scaled with max value of pixel-> 255.0. This is expected
## by all libraries
dataset_mean[0] = dataset_mean[0]/255.0
dataset_mean[1] = dataset_mean[1]/255.0
dataset_mean[2] = dataset_mean[2]/255.0

dataset_std[0] = dataset_std[0]/255.0
dataset_std[1] = dataset_std[1]/255.0
dataset_std[2] = dataset_std[2]/255.0
    
## Commenting these as this takes some time, also mean should be populated first
## and then std should be calculated
#populateMeanForTheDataset(train_path)
#populateStdForDataset(train_path)


## Above gave the following mean/std for the dataset
dataset_mean = (0.4496688992557394, 0.45574301638171105, 0.46740698326736607)
dataset_std =  (0.30231249701424134, 0.2940775677688968, 0.2954279952860252)

## Let's use fixed image processed size of 224x224 required to be fed to Deep Networks.  Many pre-trained networks
## use a fixed size image. Let's define 2 variables to control this
IMAGE_SIZE = 300
IMAGE_SIZE_YOLO = 416


# ### Data augmentation 
# This is a class whose function will be called during the data preparation, so this is part of the pipeline. Class is based on the package albumnetation which is a utility package for doing data-augmentation on the image. This package also take care of the bounding boxes transformation as those will also change after performing transformation on the image. 
class DataAugmentationTechniques():
## This class takes cares of all data-augmentation techniques at one place
## We will use to randomly pick any one or none 

    def __init__(self, IMAGE_SIZE_INPUT = IMAGE_SIZE):
        
        self.data_augment = {}
        
        ## You can add more transformations here 
        self.data_augment[0] = A.Compose([
            A.Resize(IMAGE_SIZE_INPUT, IMAGE_SIZE_INPUT),   ## Resize based on the Model Network requirement 
            A.HorizontalFlip(p=0.5),            ## Horizontal flip
            A.Rotate(15),                       ## Random Rotate
            A.Normalize(dataset_mean, dataset_std),  ## Normalize image based on the calculated mean/std before
          ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['label_index']))
        
        self.data_augment[1] = A.Compose([
            A.Resize(IMAGE_SIZE_INPUT, IMAGE_SIZE_INPUT),
            A.Rotate(15), 
            A.VerticalFlip(p=0.5),
            A.Normalize(dataset_mean, dataset_mean),
          ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['label_index']))
        
        self.data_augment[2] = A.Compose([
            A.Resize(IMAGE_SIZE_INPUT, IMAGE_SIZE_INPUT),
            A.Rotate(limit=40),
            A.RandomBrightness(limit=0.1),
            A.HorizontalFlip(),
            A.Normalize(dataset_mean, dataset_mean),
        ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['label_index']))
        
        ## Index 2 is for Resizing
        self.data_augment[3] = A.Compose([
            A.Resize(IMAGE_SIZE_INPUT, IMAGE_SIZE_INPUT),
            A.Normalize(dataset_mean, dataset_std),
        ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['label_index'] ))
        
        self.resize_normalize = A.Compose([ A.Resize(IMAGE_SIZE_INPUT, IMAGE_SIZE_INPUT),
                                           A.Normalize(dataset_mean, dataset_std)])
        
        
        
    ## This should be called after doing data-augmentation
    ## This does simple normalization by dividing by 255. Everywhere I found images
    ## those have been data-augmented have been done simple normalization
    def scale_image_and_bbox(self, image, bboxes, label):
        if(bboxes is None):
            transformed = self.resize_normalize(image=image)
            return transformed['image'], None, None
        else:
            transformed = self.data_augment[3](image=image, bboxes=[bboxes], label_index=[label])
            return transformed['image'], transformed['bboxes'][0], label
    
    ## This will crop the bbox and also take border of around 16-pixels 
    ## so that some context is still present
    def crop_based_on_bbox(self, image, bboxes, labels):
        ## First crop the image with bbox 
        x1 = bboxes[0]
        y1 = bboxes[1]
        x2 = bboxes[2] + x1
        y2 = bboxes[3] + y1
        
        x1 = x1 - 16
        y1 = y1 - 16
        if(x1 < 0 ):  ## 16-pixes context
            x1 = 0 
        if ( y1 < 0 ):
            y1 = 0
                      
        x2 = x2 + 16
        y2 = y2 + 16
        if(x2 > image.shape[1]):
            x2 = image.shape[1] - 1
        if(y2 > image.shape[0]):
            y2 = image.shape[0] - 1
        
        transform = A.Compose([
            A.Crop(x1,y1, x2,y2)],
            bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['label_index']))
        
        transformed = transform(image=image, bboxes=[bboxes], label_index=[labels])

        return transformed['image'], transformed['bboxes'][0]
    
    ## This is the function that will be called during Dataset preparation. This function randomly picks 
    ## any one transformation from the list of data_augment and apply the same to the image as input. 
    ## The transformed image, bounding box and the labels are returned as output. This function would only
    ## be called during training and not during validation or testing
    def transform_image_bbox_label(self, image, bboxes, labels, transform=None):

        if(transform is None):
            randIdx = np.random.randint(0, len(self.data_augment) -1)
            transform = self.data_augment[randIdx]
        transformed = transform(image=image, bboxes=[bboxes], label_index=[labels])
        
        scaled_img, scaled_label, scaled_bbox = transformed['image'], transformed['label_index'][0], transformed['bboxes'][0], 
        if(len(transformed['bboxes']) == 0):
            scaled_img = image
            scaled_label = labels
            scaled_bbox = bboxes

        #scaled_img = tf.cast(scaled_img/255.0, tf.float32)
        scaled_bbox = tf.cast(scaled_bbox, tf.float32)
        return scaled_img, labels, scaled_bbox
    
    
    def perform_simple_normalization(self, input_img, bbox, label):
        scaled_img, scaled_bbox, _ =  self.scale_image_and_bbox(input_img, bbox, label)
        
        scaled_img = preprocess_input(scaled_img)
        scaled_img = tf.cast(scaled_img, tf.float32)
        
        scaled_bbox = tf.cast(scaled_bbox, tf.float32)
        return scaled_img,  label, scaled_bbox
    
    ## This function simply does resize/normalize and no data-augmentation. This function will be called during
    ## testing/validation
    def resize_and_rescale(self, input_img, bbox, label):
        scaled_img, scaled_bbox, _ =  self.scale_image_and_bbox(input_img, bbox, label)
        scaled_img = tf.cast(scaled_img, tf.float32)
        if(scaled_bbox is not None):
            scaled_bbox = tf.cast(scaled_bbox, tf.float32)
        return scaled_img,  label, scaled_bbox
        
        
data_augment = DataAugmentationTechniques() 
data_augment_yolo = DataAugmentationTechniques(IMAGE_SIZE_YOLO)


# ## Loading the images and creating the metadata
# Steps:
# - Populate the data-structure DatasetMetaData which will have all information contained in place like image-path, label, encoded label, bounding-box. Final output will be array for of such objects for each image. 
# - During population it reads the all annotation files and gather information on labels, bounding box etc
# - It will also perform the sanization checks on the dataset like size of bounding box should be less the image size, labels provided are matching those of the folder names, folder names are matching those in the annotation files. 
# - Bounding box provided are in pascal format, we are converting this in coco format. 
#     
class DatasetMetaData():
    def __init__(self, base, name, file, id, label, bbox):
        # print(base, name, file)
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file
        self.id = id
        self.label = label
        self.label_encoded = None
        self.bbox = bbox
        
        ## Also prepare the scaled bounding box according to the image size
        self.scaled_bbox = None
    
    def image_name(self):
        return self.name

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    
    @staticmethod
    ## Utility function which displays the image with the bounding box 
    ## Can take bbox both in coco and pascal format
    def image_with_bbox(img, bbox, is_coco = True, figure = None, ax = None, true_bbox = None, 
                        chosen_model = None,
                        pred_label = None,
                        true_label = None
                     
                       ):
        
        if(figure is None):
            figure, ax = plt.subplots(1, figsize=(4,4))
        
        ax.imshow(img)
        
        if(bbox is not None):
            if(is_coco):
                width = bbox[2]
                height = bbox[3]
            else:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
            patch = patches.Rectangle((bbox[0], bbox[1]), 
                                      width, 
                                      height, linewidth=2, color='r', fill=False, label="Predicted Bounding Box")
        
            ax.add_patch(patch)
        
        if(pred_label is not None):
            ax.set_xlabel("Predicted = {}".format(pred_label))
            ax.set_xticklabels([])
            
        

        
        if(true_bbox is not None):
            
            if(is_coco):
                width = true_bbox[2]
                height = true_bbox[3]
            else:
                width = true_bbox[2] - true_bbox[0]
                height = true_bbox[3] - true_bbox[1]
            patch = patches.Rectangle((true_bbox[0], true_bbox[1]), 
                                      width, 
                                      height, linewidth=2, color='b', fill=False, label = "True Bounding Box")
            ax.add_patch(patch)
            plt.legend()
                        
            ax.set_ylabel("Actual = {}".format(true_label))
            ax.set_yticklabels([])


        
        if(chosen_model is not None):
            ax.set_title(f"Prediction for Model = {chosen_model}")
        
        plt.close()
        return figure
                    
    @staticmethod
    ## This is the main function which iterates over the dataset and populate the DatasetMetaData 
    def load_metadata(path, df, is_train=True):
        metadata = []
        global totalImagesDone
        anyMismatchInDirectoryName = False
        anyBadLabel = False
        directoryNameMatched= []
        directoryNameMismatched = []
        boundingBoxBigger = False
        for i in os.listdir(path):
            label_from_directory = -1
            if(i not in label_df['carname'].values):
                st.text("Label = {} not found in annotation file".format(i))
                directoryNameMismatched.append(i)
                anyMismatchInDirectoryName = True
                label_from_directory = label_df[label_df['carname'] =="Ram C/V Cargo Van Minivan 2012"].index[0]
            else:
                directoryNameMatched.append(i)
                label_from_directory = label_df[label_df['carname'] == i].index[0]
            
            for f in os.listdir(os.path.join(path, i)): 
                id = f 
                ## Read the data from data frames and populate in the object
                label = df.loc[id]['label_index']
                
                if((label != label_from_directory)):
                    st.text("Label mismatch for image = {} {} {}".format(f, label_from_directory, label))
                    anyBadLabel = True
                    
                ## Also this is a pascal format, let's convert this to coco-format 
                ## as this what expected by all Deep networks 
                bbox = (df.loc[id]['x1'], df.loc[id]['y1'], 
                        (df.loc[id]['x2'] - df.loc[id]['x1']) , (df.loc[id]['y2'] - df.loc[id]['y1']))
            
                totalImagesDone = totalImagesDone + 1
            
                if(totalImagesDone == 170):
                    updateProgressBar(1)
                    totalImagesDone = 0
                    
                metaObj = DatasetMetaData(path, i, f, id, label, bbox)
                
                img = PIL.Image.open(metaObj.image_path())
                #img = load_image(metaObj.image_path())
                #img_height = img.shape[0]
                #img_width = img.shape[1]
                img_height = img.size[1]
                img_width = img.size[0]
                
                ## Bounding box sanitization check 
                if(bbox[3] > img_height or bbox[2] > img_width): 
                    st.text("Bounding box {} larger than image size{} for image {}".format(bbox, img.size, metaObj.image_path()))
                    boundingBoxBigger = True
            
                    
                metadata.append(metaObj)
         
        ## label mismatch check      
        label_mismatch = [label  for label in label_df['carname'].values if label not in directoryNameMatched ]

        if(not anyMismatchInDirectoryName):
            st.text("All Directory name matched the annotation names, sanitization check successful!")
        else:
            st.text("Mismatch directory names in annotation file ->{}".format(label_mismatch))
            
        if(boundingBoxBigger):
            st.text("Some bounding box found to be bigger than image size")
        else:
            st.text("Success: All bounding box for the dataset found to be smaller than image size !")

        if(anyBadLabel):
            st.text("Mismatch in labels in dataset")
        else:
            st.text("Success: All labels for each image matched as that obtained from directory name!")
        return np.array(metadata)


from sklearn.preprocessing import LabelEncoder
labelEncode = None


## Top function which populates the global list metadata_train/metadata_test which will be a list of DatsetMetaData
## objects
def loadMetaDataAndLinkDataFrames():
    global metadata_train
    global metadata_test
    global labelEncode
    
    labelEncode = LabelEncoder()
    ## Load the training metadata
    
    st.write("Map Train/Test Data from annotation files and Populating Data Structures")
    
    st.text(f"\n Loading Training Data ....\n")

    metadata_train = DatasetMetaData.load_metadata(train_path, annot_train_df)
    
    
    metadata_train_label_encoded = labelEncode.fit_transform([m.label for m in metadata_train])
    metadata_train_label_encoded = to_categorical(metadata_train_label_encoded, dtype='int32')

    updateProgressBar(5)

    for idx, m in enumerate(metadata_train):
        m.label_encoded = metadata_train_label_encoded[idx]
    
    st.text(f"\n Loading Testing Data ....\n")

    updateProgressBar(5)

    metadata_test = DatasetMetaData.load_metadata(test_path, annot_test_df, False)

    metadata_test_label_encoded = labelEncode.transform([m.label for m in metadata_test])
    metadata_test_label_encoded = to_categorical(metadata_test_label_encoded, dtype='int32')

    updateProgressBar(5)

    for idx, m in enumerate(metadata_test):
        m.label_encoded = metadata_test_label_encoded[idx]
        
    updateProgressBar(5)



## This is the final function which does all pre-processing steps. This function should be called
## from the GUI event of button click as a first milestone step in the capstone project. 
def performTasksMileStone1():
    global totalImagesDone
    global progressBarValue
    global EDATaskDone
    global MetaDataLoaded
    global progressBar
    global ss
    
    st.empty()
    progressBar =  st.sidebar.progress(0)
    
    if(MetaDataLoaded):
        st.write("Data Already Mapped, skipping the tasks...")
        updateProgressBar(100)
        return
    
    totalImagesDone = 0
    progressBarValue = 0
    updateProgressBar(0)

    if(not EDATaskDone):
        st.write("Annotation files not loaded, populating the annotation files into DataFrames")
        loadTrainTestAnnotFrames()
    loadMetaDataAndLinkDataFrames()
    st.write(f"\n All Tasks for Milestone 1 Done.\n")
    MetaDataLoaded = True
    
    ss.MetaDataLoaded = MetaDataLoaded
    ss.annot_train_df = annot_train_df
    ss.annot_test_df = annot_test_df
    ss.label_df = label_df
    ss.metadata_train = metadata_train
    ss.metadata_test = metadata_test   






def plot_images(path, chosen_cat, ds_dict):
    #plot atleast few samples of a particular car class and check
  
    f, axarr = plt.subplots(5,4, figsize=(10,10))
    images_list = []
    for image in os.listdir(path):
        m = ds_dict[chosen_cat + "/" + image]
        images_list.append(m)
    for i in range(5):
        for j in range(4):
            m = images_list.pop()
            DatasetMetaData.image_with_bbox(load_image(m.image_path()), m.bbox, True, f, axarr[i,j])
    plt.close()
    return f

def PerformDisplayImage(button_clicked):
    global metadata_train
    global metadata_test
    global MetaDataLoaded
    global label_df
    global train_path
    global test_path
    global ds_dict
    
    st.empty()

    if(not MetaDataLoaded):
        performTasksMileStone1()
        
    chosen_dataset = st.sidebar.radio('Dataset', ('Train', 'Test')) 
    ds_dict = None
    
    if(chosen_dataset == 'Train'):
        ds_dict = {str(m.name+"/" + m.file):m for m in metadata_train}
    else:
        ds_dict = {str(m.name+"/" + m.file):m for m in metadata_test}

    
    random_grid = st.sidebar.checkbox('Random Grid For Category')
    chosen_image = None
    chosen_cat = None
    path = None
    
    
    if(random_grid):
        chosen_cat = st.sidebar.selectbox("Select Category", label_df['carname'].to_list())
        if(chosen_dataset == 'Train'):
            path = train_path + "/" + chosen_cat
        else:
            path = test_path + "/" + chosen_cat  
    else:
        chosen_image = st.sidebar.selectbox("Select Image", list(ds_dict.keys()))
        chosen_image = ds_dict[chosen_image]
        
    if(button_clicked):
        if(not random_grid):
            if(chosen_dataset == 'Train'):
                st.write(chosen_image.name + " --->" + chosen_image.file)
                fig = DatasetMetaData.image_with_bbox(load_image(chosen_image.image_path()), chosen_image.bbox)
                st.pyplot(fig)
            else:            
                st.write(chosen_image.name + "--->" + chosen_image.file )
                fig = DatasetMetaData.image_with_bbox(load_image(chosen_image.image_path()), chosen_image.bbox)
                st.pyplot(fig)
        else:
            st.write("Category of Cars ---->" + chosen_cat)
            f = plot_images(path, chosen_cat, ds_dict)
            st.pyplot(f)


## This function will predict the results on batch bases and not load the complete data
## This will help contain the memory so that we are not required to load the complete test
## data in-memory
from tqdm import tqdm
import sys


from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import sys


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                #buffer.write(b)
                index = 0
                found = 0
                for idx, i in enumerate(b):
                    if(i == ''):
                        index = idx
                        found = 1
                        break

                if(found):
                    buffer.write(b[:index]+"\n") #2 = buffer.getvalue()[:index]
                else:
                    buffer.write(b)
               
                
                output_func(buffer.getvalue())
                
                #sys.stdout.flush()
                #output_func(buffer.getvalue() )

                #output_func(b)
            
            old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield
        


## Define the Intersection over union , this will used as metric during bounding box evaluation. 
def IOU(y_true, y_pred):
    intersections = 0
    unions = 0
    # set the types so we are sure what type we are using

    gt = y_true
    pred = y_pred
    # Compute interection of predicted (pred) and ground truth (gt) bounding boxes
    diff_width = np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2]) - np.maximum(gt[:,0], pred[:,0])
    diff_height = np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3]) - np.maximum(gt[:,1], pred[:,1])
    intersection = diff_width * diff_height

    # Compute union
    area_gt = gt[:,2] * gt[:,3]
    area_pred = pred[:,2] * pred[:,3]
    union = area_gt + area_pred - intersection

    # Compute intersection and union over multiple boxes
    for j, _ in enumerate(union):
      if union[j] > 0 and intersection[j] > 0 and union[j] >= intersection[j]:
        intersections += intersection[j]
        unions += union[j]

    # Compute IOU. Use epsilon to prevent division by zero
    iou = np.round(intersections / (unions + tensorflow.keras.backend.epsilon()), 4)
    # This must match the type used in py_func
    iou = iou.astype(np.float32)
    return iou

def IOU_tensorflow_version(y_true, y_pred):
    iou = tensorflow.py_function(IOU, [y_true, y_pred], Tout=tensorflow.float32)
    return iou


## These are set of functions called during pipelining for dataset preparation. These functions will then called funcions
## in DataAugmentation class. 

from functools import partial
AUTOTUNE = tf.data.experimental.AUTOTUNE

## Functions to help prepare the data-pipeline.
def set_shapes(img, concat_result):
    img.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    return img, concat_result

def aug_fn(image, label, bbox):
    image_string = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    aug_img, aug_label, aug_bbox = data_augment.transform_image_bbox_label(image.numpy(), bbox, label)
    
    # You can also choose simple normalization too
    #aug_img,  aug_label, aug_bbox,  = data_augment.perform_simple_normalization(image.numpy(), bbox, label)

    ## no transformation on labels
    return aug_img, aug_bbox

def valtest_fn(image, label, bbox):
    image_string = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    ## This will only resize/rescaling and no data-augmentation, which is valid only for test/val time
    aug_img, aug_label, aug_bbox = data_augment.resize_and_rescale(image.numpy(), bbox, label)
    
    ## no transformation on labels
    return aug_img, aug_bbox


def process_data(image, dict_out, is_test_or_val = False):
    label = dict_out['class_label']
    bbox = dict_out['bounding_box']
    
    if(is_test_or_val):
        aut_img, aug_bbox = tf.numpy_function(func=valtest_fn, inp=[image, label, bbox], Tout=[tf.float32, tf.float32])
    else:
        aut_img, aug_bbox = tf.numpy_function(func=aug_fn, inp=[image, label, bbox], Tout=[tf.float32, tf.float32])
    
    ## This return the dict-based output, these names should match that used in layers
    
    ## In  case you only want to do classification just un-comment this
    #return aut_img, ({'class_label': label})
    
    ## For both bbox and classification this is needed
    return aut_img, ({'class_label' : label, 'bounding_box':aug_bbox})


def process_classify_data(image, dict_out, is_test_or_val = False):
    label = dict_out['class_label']
    bbox = dict_out['bounding_box']
    if(is_test_or_val):
        aut_img, aug_bbox = tf.numpy_function(func=valtest_fn, inp=[image, label, bbox], Tout=[tf.float32, tf.float32])
    else:
        aut_img, aug_bbox = tf.numpy_function(func=aug_fn, inp=[image, label, bbox], Tout=[tf.float32, tf.float32])
    
    ## This return the dict-based output, these names should match that used in layers
    
    ## For both bbox and classification this is needed
    return aut_img, ({'class_label' : label})


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        global progressBar
        progressBar =  st.sidebar.progress(0)
        st.write("Starting training")

    def on_train_end(self, logs=None):
        global progressBarValue
        keys = list(logs.keys())
        updateProgressBar(100)
        st.write("Stop training; ")

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        st.write("Start epoch {} of training".format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        global progressBarValue
        keys = list(logs.keys())
        updateProgressBar(5*(epoch+1))
        st.write("End epoch {} of training; got log: {}".format(epoch, logs))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        st.write("Start testing;")

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        st.write("Stop testing;")

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        #st.write("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        st.write("Stop predicting; got log : {}".format(logs))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        #st.write("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        st.write("...Training: end of batch {}; got log {}".format(batch, logs))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        #st.write("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        st.write("...Evaluating: end of batch {}; got log : {}".format(batch, logs))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        #st.write("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        st.write("...Predicting: end of batch {}; got log : {}".format(batch, logs))


## This is creating the data in batches, data creation pipe-line. This calls the data-augmentation functions provided
## as callback. 

def createDataSetForBatch(batch_size, is_classify = False):
    global metadata_train
    global metadata_test
    
    X = [m for m in metadata_train]
    Y = [m.label for m in metadata_train]

    X_train = X

    train_filelist = [m.image_path() for m in X_train]
    train_label = [m.label_encoded for m in X_train]
    train_bbox = [m.bbox for m in X_train]

    X_val = [m for m in metadata_test]
    validation_meta = X_val 
    val_filelist = [m.image_path() for m in validation_meta]
    val_label = [m.label_encoded for m in validation_meta]
    val_bbox = [m.bbox for m in validation_meta]
    
    # create train dataset using the slice list
    dataset_train = tf.data.Dataset.from_tensor_slices((train_filelist, {'class_label': train_label, 'bounding_box' : train_bbox}))    
    dataset_train = dataset_train.shuffle(len(train_filelist))
    
    if(is_classify == False):   
        dataset_train = dataset_train.map(process_data, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    else:
        dataset_train = dataset_train.map(process_classify_data, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    
    dataset_train = dataset_train.map(set_shapes, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    
    ## Create the validation dataset, this will not invoke data-augmentation pipeline, simply resize and rescale
    dataset_val = tf.data.Dataset.from_tensor_slices((val_filelist, {'class_label': val_label, 'bounding_box' : val_bbox}))
    
    #dataset_val = dataset_val.shuffle(len(val_filelist))
    if(is_classify == False):      
        dataset_val = dataset_val.map(partial(process_data, is_test_or_val=True ),  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    else:
        dataset_val = dataset_val.map(partial(process_classify_data, is_test_or_val=True ),  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)   
    
    dataset_val = dataset_val.map(set_shapes, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return dataset_train, dataset_val



def GetMobileNetV2Model():
    base_model= tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights='imagenet', 
                            include_top=False) 

    x= base_model.output
    
    for layer in base_model.layers:
        layer.trainable=False
        
    softmaxHead = tf.keras.layers.GlobalAveragePooling2D()(x)
    softmaxHead = Dense(196, activation="softmax", name="class_label")(softmaxHead)
    
    bboxHead = tensorflow.keras.layers.GlobalMaxPooling2D()(x)
    bboxHead = Dense(4, activation="relu",name="bounding_box")(bboxHead)

    ## We have 2-outputs here 
    model_combined=Model(inputs=base_model.input,outputs=[softmaxHead, bboxHead])
        
    return model_combined

## Let's prepare the EfficientNetB5, using imagenet as the weights for the classification problem
## One can edit this function to change the model for the classification model.  Rest of the things
## will remain same

from tensorflow.keras.applications.inception_v3 import InceptionV3

def GetEfficientNetB5Model_issues():
    base_model= tf.keras.applications.EfficientNetB5(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights='imagenet', 
                            include_top=False) 

    #base_model= InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights='imagenet', 
     #                       include_top=False) 

    x= base_model.output
    
    for layer in base_model.layers:
        layer.trainable=False
        
    softmaxHead = tf.keras.layers.GlobalAveragePooling2D()(x)
    softmaxHead = Dense(196, activation="softmax", name="class_label")(softmaxHead)
    
    bboxHead = tensorflow.keras.layers.GlobalMaxPooling2D()(x)
    bboxHead = Dense(4, activation="relu",name="bounding_box")(bboxHead)

    ## We have 2-outputs here 
    model_combined=Model(inputs=base_model.input,outputs=[softmaxHead, bboxHead])
        
    ## Enable the training of batch-normalization layer, there is bug in batch-normalization
    ## layer which making accuracy to be weird during the validation inference. So I am 
    ## enable the training of the batch-layer again
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
        
    return model_combined


    ## Let's prepare the EfficientNetB5, using imagenet as the weights for the classification problem
## One can edit this function to change the model for the classification model.  Rest of the things
## will remain same.  
### Previous implementation has issues with bounding box max pooling was trimming to features of the image
### and so the bounding box were all shifted to one side.  Insead of using max-pooling for bbox, instead
## flattend the layer and fed to the final output.  
from tensorflow.keras.applications.inception_v3 import InceptionV3
def GetEfficientNetB5Model():
    base_model= tf.keras.applications.EfficientNetB5(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights='imagenet', 
                            include_top=False) 

    x= base_model.output
    
    for layer in base_model.layers:
        layer.trainable=False
        
    softmaxHead = tf.keras.layers.GlobalAveragePooling2D()(x)
    softmaxHead = Dense(196, activation="softmax", name="class_label")(softmaxHead)
    
    #bboxHead = tensorflow.keras.layers.GlobalMaxPooling2D()(x)
    bboxHead = tf.keras.layers.Flatten()(x)
    bboxHead = Dense(4, activation="linear",name="bounding_box")(bboxHead)

    ## We have 2-outputs here 
    model_combined=Model(inputs=base_model.input,outputs=[softmaxHead, bboxHead])
        
    ## Enable the training of batch-normalization layer, there is bug in batch-normalization
    ## layer which making accuracy to be weird during the validation inference. So I am 
    ## enable the training of the batch-layer again
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
        
    return model_combined
   
def GetMobileNetV2ClassificationModel():
    base_model= tf.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights='imagenet', 
                            include_top=False,alpha=1.4) 

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024,activation="relu")(x)
    x = Dropout(.6)(x)
    
    softmaxHead = Dense(196, activation="softmax", name="class_label")(x)        

    ## We have 1-outputs here 
    classification_model=Model(inputs=base_model.input,outputs=[softmaxHead])
        
    return classification_model
    
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    
def TrainMobileNetV2ClassificationModel(epochs):
    #st.empty()
    batch_size = 64
    dataset_train, dataset_val = createDataSetForBatch(batch_size, True)

    # SGD optimizer
    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # Model checkpointing callback
    model_checkpoint = ModelCheckpoint("mobilenet_sgd3_-{val_loss:.2f}.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
    stop = EarlyStopping(monitor="val_loss", patience=9, mode = 'min')
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5, verbose=1, mode = 'min')
    callbacks = [model_checkpoint, reduce_lr, stop, CustomCallback()]

    classification_model =  GetMobileNetV2ClassificationModel()
    
    # Compile model
    classification_model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # Train the model to fit the training data and compare against validation set
    classification_model.fit(
        dataset_train, validation_data=dataset_val,
        callbacks=callbacks,
        epochs=epochs,
        verbose=1
    )
    
    classification_model.save_weights("mobilenet_sgd_dropout_weights.hdf5")
    classification_model.save("MobileNetClassifier_dropout_model.h5")

def ModelTraining(model, batch_size, epochs, lr, lossWeights):
    losses = {
        "class_label": "categorical_crossentropy",
        "bounding_box": "mean_squared_error",

    }

    metricsAll = {
        "class_label": tensorflow.keras.metrics.CategoricalAccuracy(),
        "bounding_box": IOU_tensorflow_version
    }

    dataset_train, dataset_val = createDataSetForBatch(batch_size)

    ## Let' use a checkpoint callback, saving weights whenever our accurarcy exceeds last reached value and
    ## only saving the max one in weights
    checkpoint = ModelCheckpoint('weights_max_accurary_epochs.hdf5', monitor='val_class_label_categorical_accuracy', verbose=1, 
                             save_best_only=True, mode='max')


    ## Reducing learning rate by certain amount if accuracy doesn't changes 
    reduce_lr = ReduceLROnPlateau(monitor="val_class_label_categorical_accuracy", 
                              factor=0.2, patience=1, threshold = 0.9,verbose=1, 
                              mode = 'max')

    callbacks_list = [checkpoint, reduce_lr]


    opt = tensorflow.keras.optimizers.Adam(learning_rate=lr)

    ## Also we will run for only 20 epochs and then fined tuned these weights
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,  metrics=metricsAll)
    history = model.fit(dataset_train, validation_data=dataset_val,
              epochs=epochs, verbose=1, callbacks=callbacks_list)

    return model

## This function will predict the results on batch bases and not load the complete data
## This will help contain the memory so that we are not required to load the complete test
## data in-memory
from tqdm import tqdm
import sys

def batchWisePrediction(model, metadata, batch_size, label_encoder, IMAGE_SIZE, data_augment):
    totalBatches = int(len(metadata) / batch_size)
    lenDataset = len(metadata)
    
    Y_pred_bbox = np.zeros((0, 4))
    Y_pred_label = np.array([])
    Y_true_label = np.array([m.label for m in metadata])
    Y_true_bbox = np.zeros((0,4))

    prog_bar = st.progress(0)
    step_size = int(totalBatches/100)
    step = 0
    progress_value = 0
    for i in tqdm(range(totalBatches)):
        if(((i+1)*batch_size + batch_size ) < lenDataset):
            metadata_batch = metadata[i*batch_size:(i*batch_size + batch_size)]
        else:
            metadata_batch = metadata[i*batch_size:lenDataset]

        X_batch = np.zeros((len(metadata_batch), IMAGE_SIZE, IMAGE_SIZE, 3))
    
        for idx, m in enumerate(metadata_batch):
            scaled_img, _, scaled_bbox = data_augment.resize_and_rescale(load_image(m.image_path()), m.bbox, m.label_encoded)
            X_batch[idx] = scaled_img
            Y_true_bbox = np.append(Y_true_bbox, scaled_bbox.numpy().reshape(1,-1), axis=0)
    
        Y_pred_label_batch, Y_pred_bbox_batch = model.predict(X_batch)
        #Y_pred_label_batch= model.predict(X_batch)
        Y_pred_bbox = np.append(Y_pred_bbox, Y_pred_bbox_batch, axis=0)
        Y_pred_label = np.append(Y_pred_label, np.argmax(Y_pred_label_batch, axis=1))
        
        step = step + 1
        if(step == step_size):
            progress_value = progress_value + 1
            if(progress_value < 100):
                prog_bar.progress(progress_value)
            step = 0
            
    prog_bar.progress(100)
    Y_pred_label = Y_pred_label.astype(np.int32)
    Y_pred_label = label_encoder.inverse_transform(Y_pred_label)
    return Y_true_label , Y_pred_label, Y_true_bbox, Y_pred_bbox

## This code has been taken from public repository: 
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

## This part will further fine tune the model by enabling training for all layers
## This will use Cyclic Learning rate, however it will start with very small learning rate 
## and upper bound is also very small
def PerformFinalTuning(model, name):
    global metadata_train
    
    ## Let's load the weights found in previous 

    ## You can skip above first part of tuning and directly start further tuning-from here 
    model.load_weights("weights_max_accurary_epochs.hdf5")
    losses = {
        "class_label": "categorical_crossentropy",
        "bounding_box": "mean_squared_error",
    }

    # define a dictionary that specifies the weights per loss (both the
    # class label and bounding box outputs will receive equal weight)
    lossWeights = {
        ## As bounding box is quite good, we can concentrate on classification
        ## and further reduce the effect of bounding-boxes
        "class_label": 10.0,
        "bounding_box": 0.01,
    }

    metricsAll = {
        "class_label": tensorflow.keras.metrics.CategoricalAccuracy(),
        "bounding_box": IOU_tensorflow_version
    }

    total_layers = len(model.layers)

    opt = tensorflow.keras.optimizers.Adam(lr=1.2799998785339995e-05)
    clr = CyclicLR(mode='triangular', base_lr=6.399999165296323e-06,  ## These are the rates obtained from LRFinder
                   max_lr=0.0001,
                   step_size= 8 * (len(metadata_train) // batch_size))

    checkpoint = ModelCheckpoint(name + 'weights_second_progunfreeze_epochs_efficientnetB5.hdf5', monitor='val_class_label_loss', verbose=1, 
                                 save_best_only=True, mode='min')


    ## Reducing learning rate by certain amount if accuracy doesn't changes 
    reduce_lr = ReduceLROnPlateau(monitor="val_class_class_label_loss", 
                                  factor=0.2, patience=1, threshold = 0.9,verbose=1, 
                                  mode = 'min')

    callbacks_list = [checkpoint, reduce_lr, clr]

    batch_size = 16
    dataset_train, dataset_val = createDataSetForBatch(batch_size)

    model.trainable = True
    for layer in model.layers:
        layer.trainable = True
    
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,  metrics=metricsAll)
    history = model.fit(dataset_train, validation_data=dataset_val, epochs=20, verbose=1, callbacks=callbacks_list)
    

# create a YOLOv3 Keras model and save it to file
# based on https://github.com/experiencor/keras-yolo3
import struct
import numpy as np
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model

 
def _conv_block(inp, convs, skip=True):
	x = inp
	count = 0
	for conv in convs:
		if count == (len(convs) - 2) and skip:
			skip_connection = x
		count += 1
		if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
		x = Conv2D(conv['filter'],
				   conv['kernel'],
				   strides=conv['stride'],
				   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
				   name='conv_' + str(conv['layer_idx']),
				   use_bias=False if conv['bnorm'] else True)(x)
		if conv['bnorm']: x = BatchNormalization(epsilon=0.00001, name='bnorm_' + str(conv['layer_idx']))(x)
		if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
	return add([skip_connection, x]) if skip else x
 
def make_yolov3_model():
    input_image = Input(shape=(IMAGE_SIZE_YOLO, IMAGE_SIZE_YOLO, 3))
    
    #input_image = Input(shape=(None, None, 3))
    # Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
        
    skip_36 = x
        
    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
        
    skip_61 = x
        
    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
        
    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)

    # Layer 80 => 82
    yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                              {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)

    # Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)

    # Layer 92 => 94

    yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                              {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)
    
    
    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
    x = UpSampling2D(2)(x)
   # x = concatenate([x, skip_36])


    yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                               {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)
    
    softmaxHead = tf.keras.layers.GlobalAveragePooling2D()(x)
   # softmaxHead = Flatten()(softmaxHead)
    softmaxHead = Dense(196, activation="softmax", name="class_label")(softmaxHead)
    
    bboxHead = tensorflow.keras.layers.GlobalMaxPooling2D()(x)
   # bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="relu",name="bounding_box")(bboxHead)
    
    model = Model(input_image, outputs=[softmaxHead, bboxHead])

    return model

def PerformModelTraining(button_clicked):
    global MetaDataLoaded
    
    st.empty()

    chosen_model = st.sidebar.selectbox("Models", ["MobileNetV2Classification", "MobileNetV2", "EfficientNetB5"])
    
    epochs = st.sidebar.slider("epochs", min_value=5, max_value=100)
    
    if(not MetaDataLoaded):
        performTasksMileStone1()
        

        batch_size = 32
        lr = 0.001
        
        if(chosen_model == "MobileNetV2Classification"):
            TrainMobileNetV2ClassificationModel(epochs)
        elif(chosen_model == "MobileNetV2"):
            st.write("Learning Rate = 0.001, Batch_size = 128")
            model = GetMobileNetV2Model()
            lossWeights = {
                "class_label": 1.0,
                "bounding_box": 0.1,
            }
            batch_size = 128
            
            with st_stdout("success"), st_stderr("info"):
                model = ModelTraining(model, batch_size, epochs, lr, lossWeights)
                
            model.save("MobileNetV2.h5")
            
        elif(chosen_model == "EfficientNetB5"):            
            st.write("Learning Rate = 0.01, Batch_size = 28")
            model = GetEfficientNetB5Model()
            lossWeights = {
                "class_label": 10.0,
                "bounding_box": 0.1,
            }
            batch_size = 28
            lr = 0.01
            
            with st_stdout("success"), st_stderr("info"):
                model = ModelTraining(model, batch_size, epochs, lr, lossWeights)
                
            PerformFinalTuning(model, "EfficientNet")
            
            
def ModelPredictionOnImage(model, image, Y_true_bbox=None, Y_true_label=None, IMAGE_SIZE_INPUT = IMAGE_SIZE, dataaug = data_augment, isOnlyClassify = False):
    global labelEncode
    
    scaled_img, _, scaled_bbox = dataaug.resize_and_rescale(image, Y_true_bbox, Y_true_label)
                                                                 
    X = tf.reshape(scaled_img, (1,IMAGE_SIZE_INPUT, IMAGE_SIZE_INPUT, 3))
    Y_pred_bbox = None
    if(isOnlyClassify):
        Y_pred_label = model.predict(X)
    else:
        Y_pred_label, Y_pred_bbox = model.predict(X)
        
    Y_pred_label_max = np.argmax(Y_pred_label[0])
    Y_pred_label_max = Y_pred_label_max.astype(np.int32)
    Y_pred_label = labelEncode.inverse_transform(np.reshape(Y_pred_label_max, (1,-1)))
    
    image_height, image_width, _ = image.shape

    if(Y_pred_bbox is not None):
        bbox = Y_pred_bbox[0]
        x0 = int(bbox[0] * image_width / IMAGE_SIZE_INPUT) # Scale the BBox
        y0 = int(bbox[1] * image_height / IMAGE_SIZE_INPUT)

        x1 = int((bbox[0] + bbox[2]) * image_width / IMAGE_SIZE_INPUT)
        y1 = int((bbox[1] + bbox[3]) * image_height / IMAGE_SIZE_INPUT)
    
        x1 = x1 - x0
        y1 = y1 - y0
        return Y_pred_label, (x0,y0,x1,y1)
    else:
        return Y_pred_label

global pre_loaded_model_path
global pre_loaded_model_objects
global dict_pre_loaded_model_weights
global dict_pre_loaded_model_creation

pre_loaded_model_path = "./"

## some how model save is not working as this doesn't save the metrics
## so loading models based on weights
dict_pre_loaded_model_creation = {"MobileNetV2": GetMobileNetV2Model,
                                  "EfficientNetB5": GetEfficientNetB5Model,
                                  "Yolo3": make_yolov3_model,
                                  "MobileNetV2Classification": GetMobileNetV2ClassificationModel
                                 }
dict_pre_loaded_model_weights = { "MobileNetV2": "weights_first_mobilev2_epochs.hdf5", 
                                 "EfficientNetB5": "weights_second_progunfreeze_epochs_efficientnetB5.hdf5",
                                 "Yolo3": "car_obj_detection_fined_tuned_yolo_training.hdf5",
                                 "MobileNetV2Classification": "mobilenet_sgd_dropout_weights.hdf5"
                                }
dict_pre_loaded_model_objects = {}

import requests

## This can download the weights on the fly from the google drive
## It just require a file-id of the google drive. So for the final
## submission we can enable this code and further this can used
## 
def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)  

import os.path
from os import path

def downloadFineTunedWeights():
    efficientNetWtID = "17tTEeJMng0f4o5GjsJRzYIVD5Q1pVp4M"
    efficientNetWtFileName = pre_loaded_model_path + 'weights_second_progunfreeze_epochs_efficientnetB5.hdf5'
    
    if(not path.exists(efficientNetWtFileName)):
        st.write("Downloading EfficientNet Pre-Trained Weights....")
        download_file_from_google_drive(efficientNetWtID, efficientNetWtFileName)

    mobileNetV2WtID = '19KZuIRzrPfOwH5tHwt78x1Dq_nUryz69'
    mobileNetV2WtFileName = pre_loaded_model_path + 'weights_first_mobilev2_epochs.hdf5'

    if(not path.exists(mobileNetV2WtFileName)):
        st.write("Downloading MobileNetV2 Pre-Trained Weights....")
        download_file_from_google_drive(mobileNetV2WtID, mobileNetV2WtFileName)
        
    yolo3WtID = "1knRyUIbzvSGbumI5o0HCdwI7VGp6o8Es"
    yolo3WtFileName = pre_loaded_model_path + 'car_obj_detection_fined_tuned_yolo_training.hdf5'
    
    if(not path.exists(yolo3WtFileName)):
        st.write("Downloading Yolo3 Pre-Trained Weights....")
        download_file_from_google_drive(yolo3WtID, yolo3WtFileName)
        
    mobileNetClassifyID = '19ZBMzzIpPCfFdOEx1Z9u5xR6PAY0_tDT'
    mobileNetClassifyName = pre_loaded_model_path + 'mobilenet_sgd_dropout_weights.hdf5'
    
    if(not path.exists(mobileNetClassifyName)):
        st.write("Downloading Mobile Net Classification Model Pre-Trained Weights....")
        download_file_from_google_drive(mobileNetClassifyID, mobileNetClassifyName)

    


### TODO: Guys this needs to be implemented 
def GetPredictedOutputForAllModels(image, chosen_model, true_label=None, true_bbox=None):
    global dict_pre_loaded_model_objects
    global label_df
    
    if(chosen_model is not "ALL"):
        model = dict_pre_loaded_model_objects.get(chosen_model, 0)
        
        if(model == 0):
            return 
        
        IMAGE_SIZE_INPUT = IMAGE_SIZE
        dataaug = data_augment
        if(chosen_model == 'Yolo3'):
            IMAGE_SIZE_INPUT = IMAGE_SIZE_YOLO
            dataaug = data_augment_yolo
            
        pred_label = None
        bbox = None
        if(chosen_model == 'MobileNetV2Classification'):
            pred_label= ModelPredictionOnImage(model, image, true_bbox, true_label, IMAGE_SIZE_INPUT, dataaug, True)
        else:
            pred_label, bbox = ModelPredictionOnImage(model, image, true_bbox, true_label, IMAGE_SIZE_INPUT, dataaug)
        pred_label_name = label_df[label_df.index == pred_label[0]]['carname'].values[0]
        
        true_label_name = None
        if(true_label is not None):
            true_label_name = label_df[label_df.index == true_label]['carname'].values[0]
        
        fig = DatasetMetaData.image_with_bbox(image, bbox, 
                                              True, None, None, true_bbox, 
                                              chosen_model,
                                              pred_label_name,
                                              true_label_name
                                             )
                
    else:
        fig, axarr = plt.subplots(2,2, figsize=(10,10))

        i = 0
        j = 0
        for name, model in dict_pre_loaded_model_objects.items():
            IMAGE_SIZE_INPUT = IMAGE_SIZE
                    
            dataaug = data_augment

            if(name == 'Yolo3'):
                IMAGE_SIZE_INPUT = IMAGE_SIZE_YOLO
                dataaug = data_augment_yolo
            
            bbox = None
            
            if(name == 'MobileNetV2Classification'):
                pred_label= ModelPredictionOnImage(model, image, true_bbox, true_label, IMAGE_SIZE_INPUT, dataaug, True)
            else:
                pred_label, bbox= ModelPredictionOnImage(model, image, true_bbox, true_label, IMAGE_SIZE_INPUT, dataaug)

            pred_label_name = label_df[label_df.index == pred_label[0]]['carname'].values[0]
            
            true_label_name = None
            if(true_label is not None):
                true_label_name = label_df[label_df.index == true_label]['carname'].values[0]
        
            _ = DatasetMetaData.image_with_bbox(image, bbox, 
                                                True, fig, axarr[i, j], true_bbox, 
                                                name,
                                                pred_label_name,
                                                true_label_name
                                               )
            j = j + 1
            if(j == 2):
                j = 0            
                i = i + 1

                
    st.pyplot(fig)
    


def PerformPrediction(button_clicked):
    global metadata_test
    global MetaDataLoaded
    global label_df
    global test_path
    global dict_pre_loaded_model_objects
    global pre_loaded_model_path
    global dict_pre_loaded_models_weights
    global dict_pre_loaded_model_creation
    global ds_dict
    global labelEncode
       
    ## Populate the models
    if(len(dict_pre_loaded_model_objects) == 0):
        st.write("Loading Pre-Trained models.....")
        ### You can comment this if you have already downloaded the weights. But this
        ## will download the weight once 
        downloadFineTunedWeights()
        for name, file in dict_pre_loaded_model_weights.items():
            model = dict_pre_loaded_model_creation[name]()
            model.load_weights(pre_loaded_model_path + file)
            dict_pre_loaded_model_objects[name] = model
        ss.dict_pre_loaded_model_objects = dict_pre_loaded_model_objects
        
    st.empty()
    st.subheader("Prediction")

    if(not MetaDataLoaded):
        performTasksMileStone1()
    
    chosen_dataset = st.sidebar.radio('Load Image', ('Test Dataset', 'Upload Test Image', "Full Test DataSet")) 
    ds_dict = None
    uploaded_file = None
    chosen_image = None
    
    true_label = None
    true_bbox = None
    
    chosen_model = st.sidebar.radio('Pre-Trained Model', ("ALL", "MobileNetV2", "EfficientNetB5", "MobileNetV2Classification", "Yolo3"))
    
    if(chosen_dataset == 'Test Dataset'):
        ds_dict = {str(m.name+"/" + m.file):m for m in metadata_test}
        chosen_image = st.sidebar.selectbox("Select Image", list(ds_dict.keys()))
        chosen_image = ds_dict[chosen_image]
    elif(chosen_dataset == "Upload Test Image"):
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        ds_dict = {str(m.name+"/" + m.file):m for m in metadata_test}

    if(button_clicked):
        image = None
        if(chosen_dataset == 'Test Dataset'):
            image = load_image(chosen_image.image_path())
            true_label = chosen_image.label
            true_bbox = chosen_image.bbox
            GetPredictedOutputForAllModels(image, chosen_model, true_label, true_bbox)
        elif uploaded_file is not None:
            image = load_image(uploaded_file)
            GetPredictedOutputForAllModels(image, chosen_model, true_label, true_bbox)
        else:
            ## Prediction for full Dataset
            if(chosen_model is not "ALL"):

                model = dict_pre_loaded_model_objects.get(chosen_model, 0)
        
                if(model == 0):
                    return 
        
                IMAGE_SIZE_INPUT = IMAGE_SIZE
                dataaug = data_augment
                if(chosen_model == 'Yolo3'):
                    IMAGE_SIZE_INPUT = IMAGE_SIZE_YOLO
                    dataaug = data_augment_yolo
                Y_true_label , Y_pred_label, Y_true_bbox, Y_pred_bbox = batchWisePrediction(model, metadata_test, 64, labelEncode,
                                                                                       IMAGE_SIZE_INPUT, dataaug)
                st.write(f"Test Accuracy = {accuracy_score(Y_true_label, Y_pred_label)}")
                st.write(f"Test IOU = {IOU(Y_true_bbox, Y_pred_bbox)}")
            else:
                st.write("Please choose a particular model for prediction of complete dataset")





def GuiAppTasks():
    global annot_train_df
    global annot_test_df
    global label_df
    global metadata_train
    global metadata_test
    global EDATaskDone
    global MetaDataLoaded
    global labelEncode
    global progressBar
    global ss
    global dict_pre_loaded_model_objects
    
    ss = SessionState.get(annot_train_df = [], 
                          annot_test_df = [],
                          label_df = [],
                          metadata_train = [],
                          metadata_test = [],
                          EDATaskDone = False,
                          MetaDataLoaded = False,                          
                          labelEncode = None,
                          dict_pre_loaded_model_objects = {}
                         )
    
    ## Restore last store values in session state
    annot_train_df = ss.annot_train_df
    annot_test_df = ss.annot_test_df
    label_df = ss.label_df
    metadata_train = ss.metadata_train
    metadata_test = ss.metadata_test
    EDATaskDone = ss.EDATaskDone
    MetaDataLoaded = ss.MetaDataLoaded
    labelEncode = ss.labelEncode
    dict_pre_loaded_model_objects = ss.dict_pre_loaded_model_objects
    
    if(st.sidebar.button('Reset All Tasks')):
        clearGUIAppTaskBools()

    chosen = st.sidebar.radio('MileStones', ('EDA', 'MapData', 'Display Images', 'Train Model', 'Prediction'))
    
    button_clicked = st.sidebar.button('Submit')
    if(button_clicked):
    
        if(chosen == "EDA"):
            PerformEDA()
        if(chosen == "MapData"):
            performTasksMileStone1()
        
            
    if(chosen == "Train Model"):
        PerformModelTraining(button_clicked)
        

    if(chosen == 'Display Images'):
        PerformDisplayImage(button_clicked)
        
    if(chosen == 'Prediction'):
        PerformPrediction(button_clicked)
        
            
    ## Checkpoint in Session State
    ss.annot_train_df = annot_train_df
    ss.annot_test_df = annot_test_df
    ss.label_df = label_df
    ss.metadata_train = metadata_train
    ss.metadata_test = metadata_test   
    ss.EDATaskDone = EDATaskDone
    ss.MetaDataLoaded = MetaDataLoaded
    ss.labelEncode = labelEncode
    ss.dict_pre_loaded_model_objects = dict_pre_loaded_model_objects

GuiAppTasks()
    

