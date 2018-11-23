import numpy as np
import h5py
import math
import random
import argparse
import pandas as pd
import time
TRIPLETS_NUMBER = 100000
arod = h5py.File('./AROD_HDF/AROD.hdf','r')
IMAGES_NUMBER = len(arod['IMAGES'])

alpha = 0.3
beta = 0.7

class Triplets():
    
    def __init__(self, a = 0.3, b = 0.7, triplets_number = 100):
        self.height = 224
        self.width = 224
        self.a = a
        self.b = b    
        self.triplets = []
        self.triplets_number = triplets_number
        self.scores = arod['SCORES'][:]
        
    def generate_triplet(self):
        is_valid = False
        while (is_valid == False):
            a_index = random.randint(0,169999)
            p_index = random.randint(0,169999)
            n_index = random.randint(0,169999)
            a = self.scores[a_index][0]
            p = self.scores[p_index][0]
            n = self.scores[n_index][0]
            is_valid = self.valid_triplet(a,p,n)        
        return a,p,n,a_index,p_index,n_index
        
        
        
    def valid_triplet(self,a, p, n):
        ap = np.abs(a - p)
        an = np.abs(a - n)
        pn = np.abs(p - n)
        if an == 0:
            an = 0.00000000001
        if pn == 0:
            pn = 0.00000000001
        if self.a > ap / an:
            return False
        if self.b < ap / an:
            return False
        if self.a > ap / pn:
            return False
        if self.b < ap / pn:
            return False
        return True
    
    
    def get_required_triplets(self,):
        valid_triplets = []
        for i in range(self.triplets_number):
            t1= time.time()
            a,p,n,a_index,p_index,n_index = self.generate_triplet()
            triplet = [a_index,p_index,n_index]
            self.triplets.append(triplet)
            if i % 500 ==0:
                t2 = time.time()
                print i,t2-t1                
                
        return self.triplets,self.scores
    

if __name__ == '__main__':
    ds = Triplets(triplets_number = 1000000)
    triplets,scores  = ds.get_required_triplets()
    df = pd.DataFrame(triplets)
    df.to_csv('triplets.csv')
