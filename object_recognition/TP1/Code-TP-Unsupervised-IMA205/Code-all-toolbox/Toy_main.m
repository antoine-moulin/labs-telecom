%% Main function for toy examples
% Author:  Pietro Gori

close all
clearvars
clc
warning off

addpath('Functions')

%% Separate Gaussian
mu1 = [2,3];
sigma1 = [3,0;0,3];
L1=100;
mu2 = [12,14];
sigma2 = [5,0;0,5];
L2=50;
data = create_gauss(mu1,sigma1,L1,mu2,sigma2,L2);

[Ypca, Upca] = pca_lecture(data(:,1:2));
plotResultsPCA(data,Ypca,Upca,'linear PCA')

sigmaKPCA=2;
[Ykpca] = Kpca_gaussian_lecture(data(:,1:2),sigmaKPCA);
plotResultsPCA(data,Ykpca,[],'K-PCA')

K=2;
[idx,C] = kmeans(data(:,1:2),K);
plotResultsKMeans(data,K,idx,C)

pause

close all
clearvars
clc
warning off

%% Overlapping Gaussian 

mu1 = [2,3];
sigma1 = [1,1.5;1.5,3]; 
L1=150;
mu2 = [5,6];
sigma2 = 2*[1,1.5;1.5,3];
L2=250;
data = create_gauss(mu1,sigma1,L1,mu2,sigma2,L2);

[Ypca, Upca] = pca_lecture(data(:,1:2));
plotResultsPCA(data,Ypca,Upca,'linear PCA')

sigmaKPCA=2;
[Ykpca] = Kpca_gaussian_lecture(data(:,1:2),sigmaKPCA);
plotResultsPCA(data,Ykpca,[],'K-PCA')

K=2;
[idx,C] = kmeans(data(:,1:2),K);
plotResultsKMeans(data,K,idx,C)

pause

close all
clearvars
clc 
warning off

%% Circles
r1=2;
r2=8;
step1=0.02;
step2=0.02;
data = create_circles(r1,step1,r2,step2);

[Ypca, Upca] = pca_lecture(data(:,1:2));
plotResultsPCA(data,Ypca,Upca,'linear PCA')

sigmaKPCA=2;
[Ykpca] = Kpca_gaussian_lecture(data(:,1:2),sigmaKPCA);
plotResultsPCA(data,Ykpca,[],'K-PCA')

K=2;
[idx,C] = kmeans(data(:,1:2),K);
plotResultsKMeans(data,K,idx,C)







