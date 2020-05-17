import tensorflow as tf 
import os
import matplotlib.pyplot as plt 
import os.path
from os import path
from zipfile import ZipFile
from scipy import misc
import numpy as np
import glob
import imageio #need to pip install


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

#def downloadEyeClosedData():
'''
train_dataset_url = "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"
folder_name = os.path.basename(train_dataset_url)
path_name = "C:/Users/hanhe/.keras/datasets/mrlEyes_2018_01.zip"
if(not path.exists(path_name)):
    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), 
                                                origin=train_dataset_url)
    print("Local copy of the dataset file: {}".format(train_dataset_fp))

else:
    print("Dataset was not downloaded because it exists already at: " + path_name)


# Create a ZipFile Object and load sample.zip in it
with ZipFile(path_name, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()
'''
rootdir = os.getcwd()
#print(rootdir)
train_images= np.zeros((15000,70,70))# [[[0]*70]*70]*5000
#print( train_images.shape)
class_names = [0, 1] # 0 for closed, 1 for open
train_labels = np.zeros(15000)
index1, index2 = 0, 0
test_labels = np.zeros(15000)
test_images = np.zeros((15000,70,70))
# so the actual dataset is a whole lot bigger, but uh do we really need that haha..
file_num = 0

# the first 25 of the 37 folders will be used for training, the rest for testing
for subdir, dirs, files in os.walk(rootdir + "\mrlEyes_2018_01"):
   # print(subdir, dirs)
    #print(file_num)
    file_num +=1
    #print(check)
    for file in files:
        # if( file!= "annotation.txt" or file != "stats_2018_01.ods"): #add annotation.txt and stats_2018_01 back later
        #     print(os.path.join(subdir, file))
        #print(file_num)

        image = imageio.imread(subdir + "\\"+ file)
        #print(image.shape)
        image = crop_center(image, 70, 70)
        if(image.shape >= (70, 70)):
            if(file_num<25):    
                index1 += 1       
                if(index1 < 15000):
                    train_labels[index1] = file[16]
                    train_images[index1] = image
            else:
                index2 +=1
                if(index1 < 15000):
                    test_labels[index2] = file[16]
                    test_images[index2] = image
print(train_labels)
print(test_labels)
print(train_images)
print (index1)
