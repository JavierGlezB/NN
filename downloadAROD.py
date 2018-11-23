import numpy as np
import pandas as pd
from PIL import Image
import random 
import matplotlib.pyplot as plt
import urllib
import cv2
import threading
import h5py
import json




with open('./RandomImages.json') as json_data:
    randomimages = json.load(json_data)['RandomImages']
 

downloadedimagespath = '/media/capptu/e6ac2520-92d9-4158-85f2-ab719a6db66d/AROD/DownloadAROD/ARODIMAGES/'

datasetsizes = {'Images':300000}
datasets = ['Images']
notFoundPhoto = cv2.resize(cv2.imread('./photo_unavailable.png'),(224,224))



import multiprocessing
from multiprocessing import Pool,Manager
import time 

processesNumber = 200
successfullyDownloadedImages = []


def downloadImage (randomimage,index,images_dict,notFoundPhoto):
    for i in range(3):
        try:
            resp = urllib.urlopen(randomimage[0])
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224,224),interpolation = cv2.INTER_CUBIC)           
            error = np.sum((image - notFoundPhoto))/(224*224*3)
            
            if (error < 5):
                image = np.zeros((10,10))
            break
        except:
            image = np.zeros((10,10))
            
    images_dict[index] = [image,randomimage]


    
    
def GetImageBatch(imageIndex, randomimages, notFoundPhoto):
    t1=time.time()
    manager = multiprocessing.Manager()
    images_dict = manager.dict()
    jobs = []
    for index in range(imageIndex,imageIndex+processesNumber):
        p = multiprocessing.Process(target=downloadImage,
                                    args=(randomimages[index],index,
                                    images_dict,notFoundPhoto))
        jobs.append(p)
        p.start()
        
    for proc in jobs:
        proc.join()
    #print return_dict.values()  
    image_values = images_dict.values()
    output = []
    for j in range (0,processesNumber):
        if np.shape(image_values[j]) != (10,10):
            output.append(image_values[j])
    t2=time.time()
    print "Finished batch downloading in: "+ str(t2-t1) +' seconds'
    return output

def appendZeros(num):
    maxdigits = 6
    digits = len (num)
    diff = maxdigits - digits
    return "0"*diff + num
    
    
    
    
    
imageIndex = 308461
datasetIndex = 299999
dataset = datasets[0]
#lset = g[dataset]
while datasetIndex <= datasetsizes[dataset]:
    t1=time.time()        
    output = GetImageBatch(imageIndex,randomimages,notFoundPhoto)
    print 'output length' + str(len(output))
    for item in range(len(output)):
        if (datasetIndex <= datasetsizes[dataset] )  and (np.shape(output[item][0]) ==(224,224,3)):    
            #lset[datasetIndex] = np.array([output[item][1][1],output[item][1][2],output[item][1][3]])
            cv2.imwrite('./ARODIMAGES/'+appendZeros(str(output[item][1][3])) +','+ str(output[item][1][1]) +','+ str(output[item][1][2])  + '.jpg',output[item][0])    
            datasetIndex = datasetIndex + 1
            if datasetIndex % 50 ==0:
                print 'ds  ' + str(datasetIndex) 
        else:
            print 'out of index'
    imageIndex = imageIndex + processesNumber 
    print 'datasetIndex ' + str(datasetIndex) 
    print 'imageIndex   ' + str(imageIndex) 
    
    print 'DatasetIndex' + str(datasetIndex)
    t2= time.time()
    print 'Batch elapsed time' +str(t2 - t1)+'s'
    
    


