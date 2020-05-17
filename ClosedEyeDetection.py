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
import wget

#need this because the images in the dataset are not all the same size, 
#so cropping them to be all a size of 70*70 pixels
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def download_dataset(rootdir):
    dataset_path = rootdir+"\\mrlEyes_2018_01"
    if(not path.exists(dataset_path)):
        url = "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"
        wget.download(url, dataset_path)
        with ZipFile(dataset_path, 'r') as zipObj:
            zipObj.extractall()
        print("Dataset was downloaded at DriverAlertSystem/mrlEyes_2018_01")
    else:
        print("Dataset was not downloaded because it already exists at DriverAlertSystem/mrlEyes_2018_01")
    

def create_model():
    rootdir = os.getcwd()
    download_dataset(rootdir)

    class_names = [0, 1] # 0 for closed, 1 for open
    #using 20000 images for training, 10000 for testing
    #the actual dataset is a lot bigger, but this seems good enough
    train_images= np.zeros((20000,70,70))
    train_labels = np.zeros(20000)
    test_labels = np.zeros(10000)
    test_images = np.zeros((10000,70,70))

    index1, index2 = 0, 0
    got_all_data = False
    for subdir, dirs, files in os.walk(rootdir + "\mrlEyes_2018_01"):
        if(got_all_data):
            break
        for file in files:
            if not (file == "annotation.txt" or file == "stats_2018_01.ods"): 
                #annotation and stats are some files that are in there for some reason so uh
                image = imageio.imread(subdir + "\\"+ file)
                image = crop_center(image, 70, 70)
                #ignoring ones which are smaller than 70*70 pixels
                if(image.shape >= (70, 70)):
                    if(index1 < 20000):
                        train_labels[index1] = file[16]
                        train_images[index1] = image
                        index1 += 1       
                    else:
                        if(index2 < 10000):
                            test_labels[index2] = file[16]
                            test_images[index2] = image
                            index2 +=1
                        else:
                            got_all_data = True

    #scaling down by 255 so its in the 0-1 range, not 0-255
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(test_images[i], cmap=plt.cm.binary)
    #     intArray =test_labels.astype(int)
    #     plt.xlabel(class_names[intArray[i]])
    # plt.show()

    #creating the layers
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(70,70)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2)
    ])

    #compiling the model
    model.compile(optimizer='adam', #adam is some algorithm
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10)

    #now testing
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    model.save('closed_eye_model') #do i need? oo i think this saves the model as a folder 
    
    print(type(model))
    
    #and using it, replace img with frame later
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    #predictions = probability_model.predict(test_images)
    img = test_images[1] # Grab an image from the test dataset.
    print(img)
    img = (np.expand_dims(img,0)) # Add the image to a batch where it's the only member.
    predictions_single = probability_model.predict(img) # where predictions_single is an array of the confidences
    print("SLDKFJSDLFKJSLDKFLDSFKJDSF")
    print(np.argmax(predictions_single[0]))
    print(predictions_single )
    print("^^^^PREDICTION SINGLE")
    print(test_labels[np.argmax(predictions_single[0])])
    #class_names = [0,1]
    print(class_names[np.argmax(predictions_single[0])])
    return model

'''
def checkForClosedEye(model):
    
    #and using it, replace img with frame later
    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])
    #predictions = probability_model.predict(test_images)
    img = model.test_images[1] # Grab an image from the test dataset.
    img = (np.expand_dims(img,0)) # Add the image to a batch where it's the only member.
    predictions_single = probability_model.predict(img) # where predictions_single is an array of the confidences
    print(np.argmax(predictions_single[0]))
    print(model.test_labels[np.argmax(predictions_single[0])])
    class_names = [0,1]
    print(class_names[np.argmax(predictions_single[0])])'''
    #here i also need to check the pixel size and do the cropping
    #and just return a bool, 1 for eye open
    #and also scale down by 255