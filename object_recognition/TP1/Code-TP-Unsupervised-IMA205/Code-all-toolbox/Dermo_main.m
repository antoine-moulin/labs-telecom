%% Main function for skin lesion images segmentation
% Author:  Pietro Gori

clc
clearvars
close all

addpath('Data/Images_dermo')
addpath('Functions')

imName='IM3.jpg'; % name of image
img = double(imread(imName))/255; % normalization
img = img(2:end-1,2:end-1,:); % crop because of artifacts on some images
img = imresize(img,0.5); % resize to decrease computational time

groundTruth = imread(strcat(imName(1:end-4), '_Segmentation.jpg'))/255; % normalize
groundTruth = groundTruth(2:end-1,2:end-1); % crop to match the image
groundTruth = imresize(groundTruth,0.5);
groundTruth = imbinarize(groundTruth);

figure
subplot(1,3,1), imshow(img), xlabel('Original Image')
subplot(1,3,2), imshow(groundTruth), xlabel('Ground truth')
subplot(1,3,3), imshow(bsxfun(@times, img, cast(groundTruth,class(img)))), xlabel('Segmented Image')

[Igray] = channelSelect(img, 'b'); % select a channel

K=3; % number of classes
[idx,C] = kmeans(Igray(:),K);
labels=reshape(idx,size(Igray)); % get one label for pixel
B = labeloverlay(Igray,labels);

figure
imshow(B)
title('Segmented Image')

% Select here best class and compare with the segmentation mask using the
% function dice and/or jaccardi

mask=labels==1;
similarity1 = dice(groundTruth,mask)
disp(['Similarity for the label 1 : ' num2str(similarity1) ])

mask=labels==2;
similarity2 = dice(groundTruth,mask)
disp(['Similarity for the label 2 : ' num2str(similarity2) ])

mask=labels==3;
similarity3 = dice(groundTruth,mask)
disp(['Similarity for the label 3 : ' num2str(similarity3) ])