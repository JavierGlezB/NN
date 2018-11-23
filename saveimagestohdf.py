import h5py
import numpy as np
import multiprocessing 
from multiprocessing import Manager
import os
import cv2

PATH = '/media/capptu/AE24C06224C02F611/ARODIMAGES/'


PROCESSES_NUMBER = 200
IMAGES = os.listdir(PATH)


def readDataset():                                                       
	try:                                                     
		f = h5py.File('AROD_HDF/AROD.hdf','r+')              
 	except:                                                  
# 		f = h5py.File('AROD_HDF/AROD.hdf','w')               
#		f.create_dataset('IMAGES', (180000,224,224,3), 'int')
#		f.create_dataset('LABELS', (180000,3), 'int')
#		f.create_dataset('SCORES', (180000,1), 'f')
		print "Error"
	return f

def readImage (index):
    imagepath = PATH+IMAGES[index]
    [Id,faves,views] = IMAGES[index].split(',')
    Id = int(Id)
    faves = int(faves)
    views = int(views.split('.')[0])
    score = faves / float(views)
    img = cv2.imread(imagepath)
    return img , Id , faves , views , score




def writeImagesOnHDF():
    f = readDataset()
    for i in range(180000):
        img , Id , faves , views , score = readImage(i)
        f['IMAGES'][i] = img
        f['LABELS'][i] = np.array([Id,faves,views]) 
        f['SCORES'][i] = score
        

        if i%100 ==0:
            print i,Id
    f.close()


#writeImagesOnHDF()





















