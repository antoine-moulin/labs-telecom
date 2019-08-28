# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:30:19 2017

@author: cagnazzo
"""
import matplotlib.pyplot as plt
import numpy as np

def coeff1Dthresh(coeffs, thr, mode='hard', verbose=0):
    out = []
    out.append(coeffs[0])       
    for levelIdx in range(1,len(coeffs)):
        if verbose:
            print 'Level %d '% (levelIdx)
        tmp = np.array(coeffs[levelIdx])
        tmp[np.abs(tmp)<=thr]=0
        if mode =='soft': 
            tmp[tmp>= thr] = tmp[tmp>= thr]-thr
            tmp[tmp<=-thr] = tmp[tmp<=-thr]+thr
        out.append(tmp)
    return out
        
        

def wtView(coeffs,title='', verbose=0):

    arr=coeffs_to_array(coeffs)
    rows,cols = arr.shape
    nLevels = len(coeffs)-1
    
    scaled_wt = np.zeros([rows,cols])
    for levIdx in range(nLevels):
        if verbose:
            print "Processing level %d %dx%d" % ( levIdx, rows, cols)
        rows=rows/2
        cols=cols/2
        
        sb = abs(arr[0:rows,cols:2*cols])
        mm = sb.min()
        MM = sb.max()
        if verbose:
            print "Max %5.2f " % MM
            print "Min %5.2f " % mm
        scaled_wt[0:rows,cols:2*cols]= 1- (sb-mm)/(MM-mm)
      
        
        sb = abs(arr[rows:2*rows,0:cols])

        mm = sb.min()
        MM = sb.max()
        if verbose:
            print "Max %5.2f " % MM
            print "Min %5.2f " % mm
        scaled_wt[rows:2*rows,0:cols]= 1- (sb-mm)/(MM-mm)

        
        sb = abs(arr[rows:2*rows,cols:2*cols])

        mm = sb.min()
        MM = sb.max()
        if verbose:
            print "Max %5.2f " % MM
            print "Min %5.2f " % mm
        scaled_wt[rows:2*rows,cols:2*cols]= 1- (sb-mm)/(MM-mm)
       
    
    if  verbose:
        print "Processing approximation %dx%d" % ( rows, cols)
    sb = arr[0:rows,0:cols]

    mm = sb.min()
    MM = sb.max()
    if verbose:
        print "Max %5.2f " % MM
        print "Min %5.2f " % mm
        print sb
    scaled_wt[0:rows,0:cols]= (sb-mm)/(MM-mm)
    
    plt.imshow(scaled_wt,cmap='gray')
    plt.title(title)
    plt.show()
    

def sbVariances(arr,nLevels,verbose=0):
    arr= np.float64(arr)
    rows,cols = arr.shape   
    vars = np.zeros(3*nLevels+1)
    for levIdx in range(nLevels):
        if verbose:
            print "Processing level %d" % levIdx
        rows=rows/2
        cols=cols/2
        
        sb = arr[0:rows,cols:2*cols]
        if verbose:
            print "Var LH %5.2f " % sb.var()
        vars[levIdx*3]= sb.var()
      
        
        sb = arr[rows:2*rows,0:cols]
        if verbose:
            print "Var LH %5.2f " % sb.var()
        vars[levIdx*3+1]= sb.var()

        
        sb = arr[rows:2*rows,cols:2*cols]
        if verbose:
            print "Var LH %5.2f " % sb.var()
        vars[levIdx*3+2]= sb.var()       
    
    sb = arr[0:rows,0:cols] 
    if  verbose:
        print "Processing approximation"
        print "Var LL %5.2f " % sb.var()
    vars[3*nLevels]= sb.var()
    return vars   

 
def sbSNR(arr,arrN,nLevels=0,verbose=0):
    
    arr= np.float64(arr)
    arrN = np.float64(arrN)
    if len(arr.shape)==2:
        rows,cols = arr.shape           
        SNR = np.zeros(3*nLevels+1)
        for levIdx in range(nLevels):
            if verbose:
                print "Processing level %d" % levIdx
            rows=rows/2
            cols=cols/2
            
            sb = arr[0:rows,cols:2*cols]
            error = sb - arrN[0:rows,cols:2*cols]
            power =   (sb**2).mean()
            err = (error**2).mean()
            tmp = 10*np.log10(power/err)
            if verbose:
                print "SNR LH %5.2f dB" % tmp
            SNR[levIdx*3]= tmp
          
            
            sb = arr[rows:2*rows,0:cols]
            error = sb - arrN[rows:2*rows,0:cols]
            power =   (sb**2).mean()
            err = (error**2).mean()
            tmp = 10*np.log10(power/err)
            if verbose:
                print "SNR HL %5.2f dB" % tmp
            SNR[levIdx*3+1]= tmp
          
    
            
            sb = arr[rows:2*rows,cols:2*cols]
            error = sb - arrN[rows:2*rows,cols:2*cols]
            power =   (sb**2).mean()
            err = (error**2).mean()
            tmp = 10*np.log10(power/err)
            if verbose:
                print "SNR HH %5.2f dB" % tmp
            SNR[levIdx*3+2]= tmp
                
        
        sb = arr[0:rows,0:cols] 
        error = sb - arrN[0:rows,0:cols]
        power =  (sb**2).mean()
        err =  (error**2).mean()
        tmp = 10*np.log10(power/err) 
        if  verbose:
            print "Processing approximation"
            print "SNR LL %5.2f dB" % tmp
        SNR[3*nLevels]= tmp
    else:
        T = arr.size
        SNR = np.zeros(nLevels+1)
        for levIdx in range(nLevels):
            
            T=T/2
            sb = arr[T:2*T]
            error = sb - arrN[T:2*T]
            power =   (sb**2).mean()
            err = (error**2).mean()
            tmp = 10*np.log10(power/err)
            if verbose:
                print "Details level %d SNR %5.2f dB" % (levIdx, tmp)
            SNR[levIdx]= tmp
                  
        sb = arr[0:T] 
        error = sb - arrN[0:T]
        power =  (sb**2).mean()
        err =  (error**2).mean()
        tmp = 10*np.log10(power/err) 
        if  verbose:
            print "Approximation SNR  %5.2f dB" % tmp
        SNR[nLevels]= tmp

    return SNR
        

def miniMax(dataSize):
    table = np.array([[3, 		0.877372 ], 
        [4, 	1.076456 ],
        [5,	 	1.276276], 
        [6, 	1.474135 ],
        [7,	1.668605],
        [8, 	1.859020],
        [9, 	2.044916],
        [10, 	2.226161 ],
        [11, 	2.402888 ],
        [12, 	2.575057 ],
        [13, 	2.742753 ],
        [14, 	2.906252],
        [15,	3.065703],
        [16, 	3.221205],
        [17,	3.373025 ],
        [18,  3.521304 ],
        [19,  3.666042 ]])
    x=table[:,0]
    y=table[:,1]
    logSize = np.log2(dataSize)
   
    if logSize<x[-1] and logSize>=x[0]: # linear interpolation
        s1 = np.uint8(np.floor(logSize))
        t1 = y[s1-3]
        t2 = y[s1-2]
        t  = t1*(1-logSize+s1)+t2*(logSize-s1)
    elif logSize>=x[-1]: #linear extrapolation
        t = y[-1] + (y[-1]-y[-2])*(logSize-x[-1])
    else: #small size
        t=y[0]
    return t
    
    
def coeffs_to_array(coeffs, verbose=0):
    nLevels = len(coeffs)    
    R,C = coeffs[len(coeffs)-1][0].shape    
    arr = np.zeros([2*R,2*C])
    rows,cols = coeffs[0].shape
    arr[0:rows,0:cols] = coeffs[0]
    if verbose:
        print 'Level %d rows %3d cols %3d' % (0,rows,cols)

    
    for levIdx in range(1,nLevels):
        rows,cols = coeffs[levIdx][0].shape
        
        if verbose:
            print 'Level %d rows %3d cols %3d' % (levIdx,rows,cols)
       
        LH = coeffs[levIdx][0]
        HL = coeffs[levIdx][1]
        HH = coeffs[levIdx][2]
        
        arr[0:rows,cols:2*cols] = LH
        arr[rows:2*rows,0:cols] = HL
        arr[rows:2*rows,cols:2*cols] = HH       

    return arr
    
    
def array_to_coeffs(arr, nLevels, verbose=0):
        
    R,C = arr.shape    
    rows = R/2**nLevels
    cols = C/2**nLevels
    coeffs = []
    coeffs.append( arr[0:rows,0:cols] )
    if verbose:
        print 'Approx rows %3d cols %3d' % (rows,cols)

    
    for levIdx in range(0,nLevels):
                        
        if verbose:
            print 'Level %d rows %3d cols %3d' % (levIdx,rows,cols)
       
        lev = []
        lev.append(arr[0:rows,cols:2*cols] )
        lev.append(arr[rows:2*rows,0:cols])
        lev.append(arr[rows:2*rows,cols:2*cols])
        coeffs.append(lev)     
        
        rows = rows*2
        cols = cols*2

    return coeffs
    
 
import cv2   
def wtViewCV(arr,nLevels=0,name='Scaled WT',verbose=0):
    rows,cols = arr.shape
    
    scaled_wt = np.zeros([rows,cols])
    for levIdx in range(nLevels):
        if verbose:
            print "Processing level %d" % levIdx
        rows=rows/2
        cols=cols/2
        
        sb = abs(arr[0:rows,cols:2*cols])
        mm = sb.min()
        MM = sb.max()
        if verbose:
            print "Max %5.2f " % MM
            print "Min %5.2f " % mm
        scaled_wt[0:rows,cols:2*cols]= 1- (sb-mm)/(MM-mm)
      
        
        sb = abs(arr[rows:2*rows,0:cols])

        mm = sb.min()
        MM = sb.max()
        if verbose:
            print "Max %5.2f " % MM
            print "Min %5.2f " % mm
        scaled_wt[rows:2*rows,0:cols]= 1- (sb-mm)/(MM-mm)

        
        sb = abs(arr[rows:2*rows,cols:2*cols])

        mm = sb.min()
        MM = sb.max()
        if verbose:
            print "Max %5.2f " % MM
            print "Min %5.2f " % mm
        scaled_wt[rows:2*rows,cols:2*cols]= 1- (sb-mm)/(MM-mm)
       
    
    if  verbose:
        print "Processing approximation"
    sb = arr[0:rows,0:cols]

    mm = sb.min()
    MM = sb.max()
    if verbose:
        print "Max %5.2f " % MM
        print "Min %5.2f " % mm
    scaled_wt[0:rows,0:cols]= (sb-mm)/(MM-mm)

        
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.imshow(name,scaled_wt)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    