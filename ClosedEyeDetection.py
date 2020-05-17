import tensorflow as tf 
from tensorflow import keras
import os
import matplotlib.pyplot as plt 
import os.path
from os import path
from zipfile import ZipFile
from scipy import misc
import numpy as np
import glob
import imageio #need to pip install
import pdb


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
start = True
# the first 25 of the 37 folders will be used for training, the rest for testing
for subdir, dirs, files in os.walk(rootdir + "\mrlEyes_2018_01"):
    if(not start):
        break
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
           # if(file_num<25):  
            if(index1 < 15000):
                train_labels[index1] = file[16]
                train_images[index1] = image
                index1 += 1       
            else:
                if(index2 < 15000):
                    test_labels[index2] = file[16]
                    test_images[index2] = image
                    index2 +=1
                else:
                    start = False
print(train_labels)
print(test_labels)
print(train_images)
print (test_labels)

#scaling down by 255 so its in the 0-1 range, not 0-255
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    intArray =test_labels.astype(int)
    plt.xlabel(class_names[intArray[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(70,70)),
    keras.layers.Dense(128, activation='relu'),# reutrns 128 nodes
    #keras.layers.Dense(10,  activation='relu'),
    keras.layers.Dense(2)
])

model.compile(optimizer='adam', #adam is some algorithm
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #uhh some performance measurement... categorical cross-entropy...
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

#now testing
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
'''
#and using it, replace img with frame later
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
img = test_images[1] # Grab an image from the test dataset.
img = (np.expand_dims(img,0)) # Add the image to a batch where it's the only member.
predictions_single = probability_model.predict(img) # where predictions_single is an array of the confidences
print(np.argmax(predictions_single[0]))
print(test_labels[np.argmax(predictions_single[0])])
print(class_names[np.argmax(predictions_single[0])])'''