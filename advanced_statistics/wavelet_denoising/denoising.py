# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:23:01 2017
Practical work on wavelet denoising
@author: cagnazzo
"""

#%% Modules
import numpy as np
import pywt
import wtTools as wtt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%% Compute and show the spectrum of h_3 and its derivatives
step = 1.0/1000
nu = np.arange(-0.5,0.5, step)
h3 = 16* (np.sin(np.pi*nu))**4
plt.plot(nu,h3), plt.title('h_3')    , plt.show()


#% compute derivatives
k=1
dh=h3
dnu = nu
while k<=4:
    dh = np.diff(dh)/step
    dnu=dnu[1:]
    print('Derivative %1d'% k)
    plt.plot(dnu,dh), plt.show()
    k=k+1

#%% Create a piecewise polynomial signal 
  
x=np.zeros(512)
N1 = 100
N2 = 200
N3 = 300
N4 = 380
N5 = 512


r1 = np.arange(0,N1)
r2 = np.arange(N1,N2)
r3 = np.arange(N2,N3)
r4 = np.arange(N3,N4)
r5 = np.arange(N4,N5)

x[r1] = r1/float(N1)
x[r2] =0.5+ ((r2-float(N1))/float(N2))**2
x[r3] = x[N2-1]
t4 = (r4-float(N3))/float(N4)
x[r4] = -80*t4**3 + 20*t4**2 + x[N2-1]
x[r5] = 0.5*(1-(r5-float(N4))/(float(N5-N4)))
x[420:423] =1
plt.plot(x), plt.title('Piecewise polynomial signal')    , plt.show()

#%% Load the DWT filter 
w = pywt.Wavelet('db4')
Npoints = 2**13
step = 1.0/Npoints
HPF = np.fft.fft(w.dec_hi,Npoints)
HPF = np.fft.fftshift(HPF)
nu = np.arange(-0.5,0.5, step)
plt.title('FR of the HPF'), plt.plot(nu,abs(HPF)), plt.show()
#% compute derivatives
k=1
dh=abs(HPF)
dnu = nu
while k<=4:
    dh = np.diff(dh)/step
    dnu=dnu[1:]
    print 'Derivative %1d'%  k
    plt.plot(dnu,dh), plt.show()
    k=k+1

#%% Compute and show  the DWT

cA3, cD3, cD2, cD1 = pywt.wavedec(x, w, level=3)
plt.title('Approximation'), plt.plot(cA3), plt.show(),
plt.title('Details 3'), plt.plot(cD3), plt.show(),
plt.title('Details 2'), plt.plot(cD2), plt.show(),
plt.title('Details 1'), plt.plot(cD1), plt.show(),

plt.hist(cD3,200)
plt.show()


#%% Noise characteristics
largeN =  2**18
sigma = .5
noiseSamples = ??? # Generate largeN samples iid from N(sigma^2, 0 ) distribution
plt.hist(noiseSamples,200)
plt.title('Noise Histogram'), plt.show()
print 'Sample STD %5.2f'%  noiseSamples.std()

nLevel = 3
noiseWT = pywt.wavedec(????, w, mode='per', level=????)
noiseWTarray = np.zeros_like(noiseSamples)
start=0
for idxLevel in range (nLevel+1):
    size =  len(noiseWT[idxLevel])
    noiseWTarray[start:start+size] = ????
    start = size+start
plt.hist(noiseWTarray,200)
plt.title('Noise  DWT Histogram'), plt.show()
print 'DWT coeff  STD %5.2f'%  noiseWTarray.std()



#%%  Add noise to the 1-D signal
sigma = .1
noise = sigma*np.random.randn(x.size)
y =  x + noise
plt.title('Noisy signal'), plt.plot(y), plt.show()
print "SNR: %5.2f dB"%  wtt.sbSNR(x,y) 

#%% Denoising with universal threshold 

nLevel = 3
thr  = ????? 
coeffs = pywt.wavedec(y, w, mode='per', level=nLevel)
coeffsT = wtt.coeff1Dthresh(coeffs, thr)
xhat   = pywt.waverec(coeffsT, w, mode='per')
plt.title('Denoised signal'), plt.plot(xhat), plt.show()
print "SNR: %5.2f dB"%  wtt.sbSNR(x,????) 
#%%

#% Wavy signal
N = 512.
r1 = np.arange(0,N)
omega = r1/N/50
x  = np.sin(np.pi*r1/N) * np.sin(2*np.pi*omega*r1)
plt.plot(r1,x), plt.title('Wavy signal'), plt.show()

#  Add noise to the 1-D signal
sigma = 0.1
noise = sigma*np.random.randn(x.size)
y =  x + noise
plt.title('Noisy signal'), plt.plot(y), plt.show()
print "SNR: %5.2f dB"%  wtt.sbSNR(x,y) 

#%% Universal threshold , hard thresholding
nLevel = 4
thr  = ???
coeffs = pywt.wavedec(y, w, mode='per', level=nLevel)
coeffsT = wtt.coeff1Dthresh(coeffs, thr)
xhat   = pywt.waverec(coeffsT, w, mode='per')
plt.title('Denoised signal'), plt.plot(xhat), plt.show()
print "SNR Hard, univ: %5.2f dB"% ????

#%% Minimax threshold , hard thresholding
thrMinimax = sigma*wtt.miniMax(x.size *(1- 2 **(-nLevel)))
coeffs = pywt.wavedec(y, w, mode='per', level=nLevel)
coeffsS = wtt.coeff1Dthresh(coeffs, thrMinimax)
xhat   = pywt.waverec(coeffsS, w, mode='per')
plt.title('Denoised signal'), plt.plot(xhat), plt.show()
print "SNR: Hard, Minimax %5.2f dB"%  wtt.sbSNR(x,xhat) 


#%% universal threshold , soft thresholding
nLevel = 4
thr  = sigma * np.sqrt(2 * np.log( x.size*(1-2**(-nLevel)) ))
coeffs = pywt.wavedec(y, w, mode='per', level=nLevel)
coeffsS = wtt.coeff1Dthresh(coeffs, thr, mode='soft')
xhat   = pywt.waverec(coeffsS, w, mode='per')
plt.title('Denoised signal'), plt.plot(xhat), plt.show()
print "SNR Soft, Univ: %5.2f dB"%  wtt.sbSNR(x,xhat) 


#%% Minimax threshold , soft thresholding
thrMinimax = ????
coeffs = pywt.wavedec(y, w, mode='per', level=nLevel)
coeffsS = wtt.coeff1Dthresh(coeffs, ??? , ????)
xhat   = pywt.waverec(coeffsS, w, mode='per')
plt.title('Denoised signal'), plt.plot(xhat), plt.show()
print "SNR Soft, Minimax: %5.2f dB"%  wtt.sbSNR(x,xhat) 


#%% Read an image and add noise
img = mpimg.imread('lena.jpg')
rows,cols = np.shape(img)
sigma = 10 
noise = sigma*np.random.randn(rows,cols)
noisyImg = np.float64(img)+noise
noisySNR = wtt.sbSNR(img,noisyImg)                    

#%% Show original and noisy images
plt.imshow(img)
plt.set_cmap('gray')
plt.title('Original image')
plt.show()

plt.imshow(np.uint8(np.clip(noisyImg,0,255)))
plt.set_cmap('gray')
plt.title('Noisy image')
plt.show()


#%% Forward  DWT
wav = 'db3'
NLEV  = 4
filter_bank = pywt.Wavelet(wav)
coeffs = pywt.wavedec2(img, filter_bank, mode='per', level=NLEV)
arr = wtt.coeffs_to_array(coeffs)
wtt.wtView(coeffs, 'Original image, scaled DWT coeffs')
plt.imshow(np.abs(arr/arr.max()))
plt.set_cmap('jet')
plt.title('Unscaled DWT coefficients')
plt.show()

#%% Inverse DWT
decoded = pywt.waverec2(coeffs, filter_bank, mode='per')
plt.imshow(np.uint8(decoded))
plt.title('Inverse DWT') ,  plt.set_cmap('gray'), plt.show()

#%% DWT of noisy image
coeffsN = pywt.wavedec2(noisyImg, filter_bank, mode='per', level=NLEV)
arrN = wtt.coeffs_to_array(coeffsN)
wtt.wtView(coeffsN,'Noisy DWT')
SNR = wtt.sbSNR(arr,arrN,NLEV,1)

#%% Noise estimation
HH= arrN[????, ???]

plt.imshow((HH-HH.min())/(HH.max()-HH.min()))
plt.title('HH band of the noisy image')
plt.show() 
sigmaHat=np.median(np.abs(HH))/0.67449
print "STD estimation: %5.2f\n" % sigmaHat

#%% Denoising


#%% Asymptotic thresdhold
visu = sigmaHat * np.sqrt(2*np.log(img.size))

#%% Denoising performance bounds (knowing the ground truth)

#%% Hard thresholding
values  = np.arange(0,visu*1.2,visu*1.2/50)
SNR_hard =np.zeros_like(values,dtype='float')
for ind,thrsh in enumerate(values): 
    arrT=pywt.thresholding.hard(arrN,???)
    arrT[rows/(2**NLEV),cols/(2**NLEV)]= arrN[???] #  LL  is not thresholded
    co2 = wtt.array_to_coeffs(arrT,NLEV)
    denoised = pywt.waverec2(???, filter_bank, mode='per')
    SNR_hard[ind] = wtt.sbSNR(img,????,0,0)
    #print  thrsh, "SNR %5.2f"% SNR_hard[ind]
 
#%% Soft thresholding
values  = values  = np.arange(0,visu*1.2,visu*1.2/50)
SNR_soft =np.zeros_like(values,dtype='float')
for ind,thrsh in enumerate(values): 
    arrT=pywt.thresholding.soft(arrN,thrsh)
    arrT[rows/(2**NLEV),cols/(2**NLEV)]= arrN[rows/(2**NLEV),cols/(2**NLEV)]
    co2 = wtt.array_to_coeffs(???,NLEV)
    denoised = pywt.waverec2(co2, filter_bank, mode='per')
    SNR_soft[ind] = wtt.sbSNR(img,denoised,0,0)
    #print  thrsh, "SNR %5.2f"% SNR_soft[ind] 

#%% Show reults
plt.plot(values,SNR_hard,label='Hard')
plt.plot(values,SNR_soft,label='Soft')
plt.legend()
plt.ylabel('SNR [dB]')
plt.xlabel('Threshold')
plt.show()
#%%
print ('Estimated noise std %5.2f' % sigmaHat)
print ('Noisy image SNR %5.2f dB' % noisySNR)

print ('Denoising with wavelet %s ' % wav, ' Levels=%2d' % NLEV)
print ('Max SNR Soft %5.2f' % SNR_soft.max(),  'dB for th=%3d' % values[SNR_soft.argmax()])
print ('Max SNR Hard %5.2f' % SNR_hard.max(),  'dB for th=%3d' % values[SNR_hard.argmax()])




#%%  Universal threshold
thrsh =????
print ('Universal threshold: %5.2f'%  thrsh)
arrT=pywt.thresholding.soft(arrN,thrsh)
arrT[rows/(2**NLEV),cols/(2**NLEV)]= arrN[rows/(2**NLEV),cols/(2**NLEV)]
co2 = wtt.array_to_coeffs(arrT,NLEV)
denoised = pywt.waverec2(co2, filter_bank, mode='per')
print ('Universal threshold Soft SNR %5.2f' % wtt.sbSNR(img,denoised))
plt.imshow(denoised)
plt.show()
arrT=pywt.thresholding.hard(arrN,thrsh)
arrT[rows/(2**NLEV),cols/(2**NLEV)]= arrN[rows/(2**NLEV),cols/(2**NLEV)]
co2 = wtt.array_to_coeffs(arrT,NLEV)
denoised = pywt.waverec2(co2, filter_bank, mode='per')
print ('Universal threshold Hard SNR %5.2f' % wtt.sbSNR(img,denoised))
plt.imshow(denoised)
plt.show()

#%% Minimax
thrsh = sigmaHat *  wtt.miniMax(img.size) 
print ('Minimax threshold: %5.2f'%  thrsh)
arrT=pywt.thresholding.soft(arrN,thrsh)
arrT[rows/(2**NLEV),cols/(2**NLEV)]= arrN[rows/(2**NLEV),cols/(2**NLEV)]
co2 = wtt.array_to_coeffs(arrT,NLEV)
denoised = pywt.waverec2(co2, filter_bank, mode='per')
plt.imshow(denoised)
plt.show()
print ('Minimax Soft SNR %5.2f' % wtt.sbSNR(img,denoised))
arrT=pywt.thresholding.hard(arrN,thrsh)
arrT[rows/(2**NLEV),cols/(2**NLEV)]= arrN[rows/(2**NLEV),cols/(2**NLEV)]
co2 = wtt.array_to_coeffs(arrT,NLEV)
denoised = pywt.waverec2(co2, filter_bank, mode='per')
print ('Minimax Hard SNR %5.2f' % wtt.sbSNR(img,denoised))
plt.imshow(denoised)
plt.show()

#%% SURE
import sure

arrT= sure.hybridDenoise(arrN,sigmaHat,NLEV) 
co2 = wtt.array_to_coeffs(arrT,NLEV)
denoised = pywt.waverec2(co2, filter_bank, mode='per')
print ('SURE SNR %5.2f' % wtt.sbSNR(img,denoised))
plt.imshow(denoised)
plt.show()

#%% Wiener 
y = noisyImg
Y = np.fft.fft2(y-y.mean())
S = Y*np.conjugate(Y) / Y.size

H = np.real( (S-sigmaHat**2)/(S+1e-14))
H =H*(H>0)
X = H*Y
denoised =np.real(np.fft.ifft2(X))+y.mean()
print ('Wiener SNR %5.2f' % wtt.sbSNR(img,denoised))
plt.imshow(denoised)
plt.show()

#%% Undecimanted WT


R,C = noisyImg.shape
acc = np.zeros_like(noisyImg,dtype='float')
shifts =  5 #2**NLEV
print '---- shifts %dx%d' % (shifts,shifts)
for dx in range(shifts+1):
    for dy in range(shifts+1):
        #print dx, dy
        RR = np.mod(np.arange(R) +dx,R)
        CC = np.mod(np.arange(C) +dy,C)
        tmp =  noisyImg[np.meshgrid(RR, CC)]
        coeffsN = pywt.wavedec2(tmp, filter_bank, mode='per', level=NLEV)
        arrN = wtt.coeffs_to_array(coeffsN)
        arrT= sure.hybridDenoise(arrN,sigmaHat,NLEV) 
        co2 = wtt.array_to_coeffs(arrT,NLEV)
        denoised = pywt.waverec2(co2, filter_bank, mode='per')

        RR = np.mod(np.arange(R) +R-dx,R)
        CC = np.mod(np.arange(C) +C-dy,C)
        den =  denoised[np.meshgrid(CC, RR)]         
        
        acc = acc+ den/((shifts+1)**2)

print ('UDWT SNR %5.2f' % wtt.sbSNR(img,acc))
plt.imshow(acc)
plt.show()       
        