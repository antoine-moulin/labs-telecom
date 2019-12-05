%% Main function for face recongition
% Author:  Pietro Gori

close all
clearvars
clc

rng('default'); % for repeteability

addpath('Functions')
addpath('Data/Face')


%% Parameters
% taken from http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html
load('YaleB_32x32.mat'); 
data=fea;
maxValue = max(max(data));
data = data/maxValue; % Scale pixels to [0,1]
clear fea;

Ns=length(unique(gnd)); % Number subjects
Is=round(length(gnd)/Ns); % Number images per subject (on average, not the same number for every subject)
r=sqrt(size(data,2)); % number rows of each image
c=sqrt(size(data,2)); % number columns of each image

index=1:Is:Is*40;

%% Plot data
figure
subplot(5,2,1)
imshow(reshape(data(index(13),:)',r,c))
subplot(5,2,2)
imshow(reshape(data(index(8),:),r,c))
subplot(5,2,3)
imshow(reshape(data(index(1),:),r,c))
subplot(5,2,4)
imshow(reshape(data(index(22),:),r,c))
subplot(5,2,5)
imshow(reshape(data(index(25),:),r,c))
subplot(5,2,6)
imshow(reshape(data(index(26),:),r,c))
subplot(5,2,7)
imshow(reshape(data(index(29),:),r,c))
subplot(5,2,8)
imshow(reshape(data(index(32),:),r,c))
subplot(5,2,9)
imshow(reshape(data(index(34),:),r,c))
subplot(5,2,10)
imshow(reshape(data(index(37),:),r,c))
suptitle('Ten different subject')

figure
subplot(5,2,1)
imshow(reshape(data(1,:),r,c))
subplot(5,2,2)
imshow(reshape(data(2,:),r,c))
subplot(5,2,3)
imshow(reshape(data(3,:),r,c))
subplot(5,2,4)
imshow(reshape(data(4,:),r,c))
subplot(5,2,5)
imshow(reshape(data(5,:),r,c))
subplot(5,2,6)
imshow(reshape(data(6,:),r,c))
subplot(5,2,7)
imshow(reshape(data(7,:),r,c))
subplot(5,2,8)
imshow(reshape(data(8,:),r,c))
subplot(5,2,9)
imshow(reshape(data(9,:),r,c))
subplot(5,2,10)
imshow(reshape(data(10,:),r,c))
suptitle('Ten positions of the same subject')

%% Code
partition='Random_partition.mat'; 
load(partition) 
Xtest=data(testIdx,:);
Xctest=Xtest-repmat(mean(Xtest),size(Xtest,1),1); % centering
Xtrain=data(trainIdx,:);
Xctrain=Xtrain-repmat(mean(Xtrain),size(Xtrain,1),1); % centering
Id_Train = gnd(trainIdx);
Id_Test = gnd(testIdx); 

[N,d]=size(Xctrain); % N number of images, d number of pixels
[M,~]=size(Xctest); % M sumber of images in the test set

%% Using simply the pixel intensities

[resultPixel,correctPixel] = nearest_neighbor(Xctrain,Id_Train,Xctest,Id_Test);
disp(['Percentage of correct answer using the pixel intensities is ' num2str(resultPixel) ])

%% PCA
disp('PCA')
tic
[Ytrain_PCA,Utrain,~,var_explained] = pca_lecture(Xctrain);

% Threshold defined as 99% of the variability
Threshold_PCA = 99 ;
Cumulative=cumsum(var_explained);
index=find(Cumulative>Threshold_PCA);
PCAComp=index(1);

% Selection of the eigenvectors 
Yr_train_PCA=Ytrain_PCA(:,1:PCAComp);
Ur_train_PCA=Utrain(:,1:PCAComp);

% Computation of the test scores using the eigenvectors computed with the
% training data-set
Yr_test_PCA=Xctest*Ur_train_PCA;

% Plot the Eigenfaces

figure
subplot(5,2,1)
imagesc(reshape(Utrain(:,1),r,c))
colormap(gray)
subplot(5,2,2)
imagesc(reshape(Utrain(:,2),r,c))
colormap(gray)
subplot(5,2,3)
imagesc(reshape(Utrain(:,3),r,c))
colormap(gray)
subplot(5,2,4)
imagesc(reshape(Utrain(:,4),r,c))
colormap(gray)
subplot(5,2,5)
imagesc(reshape(Utrain(:,5),r,c))
colormap(gray)
subplot(5,2,6)
imagesc(reshape(Utrain(:,6),r,c))
colormap(gray)
subplot(5,2,7)
imagesc(reshape(Utrain(:,7),r,c))
colormap(gray)
subplot(5,2,8)
imagesc(reshape(Utrain(:,8),r,c))
colormap(gray)
subplot(5,2,9)
imagesc(reshape(Utrain(:,9),r,c))
colormap(gray)
subplot(5,2,10)
imagesc(reshape(Utrain(:,10),r,c))
colormap(gray)
%suptitle('PCA - Eigenfaces')

[resultPCA,correctPCA] = nearest_neighbor(Yr_train_PCA,Id_Train,Yr_test_PCA,Id_Test);
disp(['Percentage of correct answer in PCA is ' num2str(resultPCA) ])

toc

%% Kernel-PCA
disp('Kernel PCA')
tic

sigma_KPCA=13; % try to change it !
[Ytrain_KPCA,An_train,~,var_explained_KPCA] = Kpca_gaussian_lecture(Xctrain,sigma_KPCA);
%exposant = 3;
%[Ytrain_KPCA,An_train,~,var_explained_KPCA] = Kpca_poly_lecture(Xctrain,exposant);

% Threshold defined as 99% of the variability
Threshold_KPCA = 99 ;
Cumulative=cumsum(var_explained_KPCA);
index=find(Cumulative>Threshold_KPCA);
KPCAComp=index(1);

% Selection of the eigenvectors 
Yr_train_KPCA=Ytrain_KPCA(:,1:KPCAComp);
Anr_train_KPCA=An_train(:,1:KPCAComp);

% Construction matrix K for test
NormTrain2 = repmat(sum(Xctrain.^2, 2)',M,1);
NormTest2 = repmat(sum(Xctest.^2, 2),1,N);
InnerX = Xctest * Xctrain';
Dtest = NormTrain2 + NormTest2 - 2*InnerX;

Ktest=exp(-Dtest./(2*sigma_KPCA^2));    
%Ktest = InnerX .^ exposant;

% Centering kernel test matrix
oneN=ones(N,N)/N;
oneM=ones(M,M)/M;
Ktestc=Ktest-oneM*Ktest-Ktest*oneN+oneM*Ktest*oneN; % center kernel matrix

% Computation of the test scores using the eigenvectors computed with the
% training data-set
Yr_test_KPCA = Ktestc * Anr_train_KPCA;

[resultKPCA,correctKPCA] = nearest_neighbor(Yr_train_KPCA,Id_Train,Yr_test_KPCA,Id_Test);
disp(['Percentage of correct answer in KPCA is ' num2str(resultKPCA) ])
toc


%% ICA - Independent component analysis
disp('ICA')
tic

% Instead than using the original data, we use the scores of PCA to reduce
% the computational time
[Ytrain_PCA,Utrain,~,var_explained] = pca_lecture(Xctrain);
Threshold_PCA = 99 ;
Cumulative=cumsum(var_explained);
index=find(Cumulative>Threshold_PCA);
PCAComp=index(1);
Yr_train_PCA=Ytrain_PCA(:,1:PCAComp);
Ur_train_PCA=Utrain(:,1:PCAComp);

[S_train_ICA, W_train_ICA] = fastICA_lecture(Yr_train_PCA',[],[],1); % be careful ICA is made for a matrix [d x N]
R=Ur_train_PCA*W_train_ICA'; 
Yr_test_PCA=Xctest*Ur_train_PCA;
% we transpose simply because the function nearest_neighbor accepts as
% input a matrix with [N_sample,dim]
Btrain_ICA=S_train_ICA'; 
Btest_ICA=W_train_ICA*Yr_test_PCA';
Btest_ICA=Btest_ICA';

[resultICA,correctICA]  = nearest_neighbor(Btrain_ICA,Id_Train,Btest_ICA,Id_Test);
disp(['Percentage of correct answer in ICA is ' num2str(resultICA) ])


figure
subplot(5,2,1)
imagesc(reshape(R(:,1),r,c))
colormap(gray)
subplot(5,2,2)
imagesc(reshape(R(:,2),r,c))
colormap(gray)
subplot(5,2,3)
imagesc(reshape(R(:,3),r,c))
colormap(gray)
subplot(5,2,4)
imagesc(reshape(R(:,4),r,c))
colormap(gray)
subplot(5,2,5)
imagesc(reshape(R(:,5),r,c))
colormap(gray)
subplot(5,2,6)
imagesc(reshape(R(:,6),r,c))
colormap(gray)
subplot(5,2,7)
imagesc(reshape(R(:,7),r,c))
colormap(gray)
subplot(5,2,8)
imagesc(reshape(R(:,8),r,c))
colormap(gray)
subplot(5,2,9)
imagesc(reshape(R(:,9),r,c))
colormap(gray)
subplot(5,2,10)
imagesc(reshape(R(:,10),r,c))
colormap(gray)
suptitle('ICA')

toc

%% NNMF
disp('NNMF - Non-negative Matrix factorization')
tic
[W_train_nnmf,Htrain_nnmf] = nnmf_lecture(Xtrain',100,500,1e-5,1); % be careful nnmf is made for a matrix [d x N]
Htest_nnmf=W_train_nnmf\Xtest';

[resultNNMF,correctNNMF]  = nearest_neighbor(Htrain_nnmf',Id_Train,Htest_nnmf',Id_Test);
disp(['Percentage of correct answer in NNMF is ' num2str(resultNNMF) ])


figure
subplot(5,2,1)
imagesc(reshape(W_train_nnmf(:,1),r,c))
colormap(gray)
subplot(5,2,2)
imagesc(reshape(W_train_nnmf(:,2),r,c))
colormap(gray)
subplot(5,2,3)
imagesc(reshape(W_train_nnmf(:,3),r,c))
colormap(gray)
subplot(5,2,4)
imagesc(reshape(W_train_nnmf(:,4),r,c))
colormap(gray)
subplot(5,2,5)
imagesc(reshape(W_train_nnmf(:,5),r,c))
colormap(gray)
subplot(5,2,6)
imagesc(reshape(W_train_nnmf(:,6),r,c))
colormap(gray)
subplot(5,2,7)
imagesc(reshape(W_train_nnmf(:,7),r,c))
colormap(gray)
subplot(5,2,8)
imagesc(reshape(W_train_nnmf(:,8),r,c))
colormap(gray)
subplot(5,2,9)
imagesc(reshape(W_train_nnmf(:,9),r,c))
colormap(gray)
subplot(5,2,10)
imagesc(reshape(W_train_nnmf(:,10),r,c))
colormap(gray)
suptitle('NNMF')

toc






