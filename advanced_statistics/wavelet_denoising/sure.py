# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:24:09 2017

@author: cagnazzo
"""

import numpy as np
import pywt
#%% Quicksort


#%% SURE threshold determination

def sureThresh(x,verbose=0):    
    t=np.sort(np.abs(np.float32(np.reshape(x,(1,x.size)))))     
    a=t**2
    b=np.cumsum(a)
    c= np.arange(x.size-1,-1,-1)
    s=b+c*a
    risk = (x.size - (np.arange(2,2*x.size+2,2)) +s)/x.size
    return t[0,np.argmin(risk)]
       

def hybridThresh(x,verbose=0):
    R,C = x.shape     
    y = np.reshape(x,(1,R*C)) 
    n = y.size
    J = np.log2(n)
    magic = np.sqrt(2*np.log(n))
    normYsq =  np.sum(y**2) 
    eta = (normYsq - n)/n

    if eta < J**(1.5)/np.sqrt(n):
        T=magic
    else:
        T=min(sureThresh(y,verbose),magic)
        
    xhat = pywt.thresholding.soft(y,T) 
    
    return np.reshape(xhat,(R,C)) 
            
def hybridDenoise(x,sigmaHat, nLevels=0,verbose=0):
    arr=np.float64(x) / sigmaHat
    rows,cols = arr.shape
    
    denoised = np.zeros([rows,cols])
    for levIdx in range(nLevels):
        if verbose:
            print "Processing level %d" % levIdx
        rows=rows/2
        cols=cols/2
        
        sb = arr[0:rows,cols:2*cols]
        denoised[0:rows,cols:2*cols]= hybridThresh(sb,verbose)
      
        
        sb = arr[rows:2*rows,0:cols]
        denoised[rows:2*rows,0:cols]= hybridThresh(sb,verbose)

        
        sb = arr[rows:2*rows,cols:2*cols]
        denoised[rows:2*rows,cols:2*cols]=hybridThresh(sb,verbose)
       
    
    if  verbose:
        print "Processing approximation" 
    denoised[0:rows,0:cols]=arr[0:rows,0:cols]
    
    return denoised*sigmaHat
         