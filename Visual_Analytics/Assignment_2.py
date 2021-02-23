#!/usr/bin/env python
# coding: utf-8

# ## Assignment 2 Description 

# 
# Download the Oxford-17 flowers image data set, available at this link:
# 
# 
# https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
# 
# 
# Choose one image in your data that you want to be the 'target image'. Write a Python script or Notebook which does the following:
# 
# Use the cv2.compareHist() function to compare the 3D color histogram for your target image to each of the other images in the corpus one-by-one.
# In particular, use chi-square distance method, like we used in class. Round this number to 2 decimal places.
# Save the results from this comparison as a single .csv file, showing the distance between your target image and each of the other images. The .csv file should show the filename for every image in your data except the target and the distance metric between that image and your target. Call your columns: filename, distance.
# Print the filename of the image which is 'closest' to your target image
# 
# Purpose
# 
# This assignment is designed to test that you have a understanding of:
# 
# how to make extract features from images based on colour space;
# how to compare images for similarity based on their colour histogram;
# how to combine these skills to create an image 'search engine'

# __Dependencies__ 

# In[18]:


#Load in the necessary packages
import os
import sys
sys.path.append(os.path.join(".."))
from pathlib import Path 
import argparse 

#impage processing
import cv2
import numpy as np
import pandas as pd

#display 
from utils.imutils import jimshow
import matplotlib.pyplot as plt


# __Create the path to the flower images__

# In[19]:


path = os.path.join("..", "data", "Flowers")
target = "image_1360.jpg"


# __Define the function to loop through images and make comparisons__

# In[24]:


def compare_images(image_directory, target_name):
    
    # read in the target image 
    target_image = cv2.imread(os.path.join(image_directory, target_name))
    # create histogram for all 3 color channels
    target_hist = cv2.calcHist([target_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
    # normalise the histogram - We do this to get rid of outliers and make the comparisons a bit more robust
    target_hist_norm = cv2.normalize(target_hist, target_hist, 0,255, cv2.NORM_MINMAX)
    
    # Create an empty dataframe for the information to be saved into  
    data = pd.DataFrame(columns=["filename", "distance"])
    
    # Now our loop begins. We're saying... for each image (ending with .jpg) in the directory
    for image_path in Path(image_directory).glob("*.jpg"):
        # take only the image name
        _, image = os.path.split(image_path)
        # if this image is not the target image
        if image != target_name:
            # then read the image and save it as a comparison image
            comparison_image = cv2.imread(os.path.join(image_directory, image))
            # Next we want to create a histogram for comparison iamge
            comparison_hist = cv2.calcHist([comparison_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
            # Normalise this comparison image 
            comparison_hist_norm = cv2.normalize(comparison_hist, comparison_hist, 0,255, cv2.NORM_MINMAX)    
            # and calculate the chisquare distance to tell how far apart they are - smaller numbers are closer 
            distance = round(cv2.compareHist(target_hist_norm, comparison_hist_norm, cv2.HISTCMP_CHISQR), 2)
            # Finally, we append this info to the dataframe
            data = data.append({"filename": image, 
                                "distance": distance}, ignore_index = True)
            
    # find image which is closest 
    print(data[data.distance == data.distance.min()]) 
    
    # save as csv in current directory (Notebooks)
    data.to_csv(f"{target_name}_comparison.csv")
    # print that file has been saved - this lets us know that it's worked 
    print(f"output file is saved in current directory as {target_name}_comparison.csv")


# In[25]:


compare_images(path, target) #Nice we find the image which is closest - it's image 1279 


# In[ ]:




