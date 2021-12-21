clear; clc; close all;

%% 
directory = "C:\Users\Owner\OneDrive - Texas Tech University\Multivariate Signal Processing\CNN code - part 2\";
resizeRatio = [227 227]; % 1.50 ratio (r/c)
%% 
files = dir(directory + "cataract type/" +"Capsular/"+ "*.jpg");
%files = dir(fullfile(directory, '*.jpg'));      

%% Train CNN - Cateract
outputSize=[227 227];
numImageCategories  = 2;
digitDatasetPath = directory + "cataract type/";
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders',true,'LabelSource','foldernames');
%% Resizing All Images into same Sizes
imds.ReadFcn = @customreader;
reset(imds);

%% CNN Design

%auimds = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',augmenter)
labelCount = countEachLabel(imds);
numClasses=height(labelCount);
numImageTraining=numel(imds.Files);

%%
numTrainFiles = 56;
[imdsTrain, imdsTest] = splitEachLabel(imds, numTrainFiles,  'randomize');
%%
imageSize=[227 227 3];

%% Convolutional layer parameters
layers1= [
    imageInputLayer(imageSize)
    %middle Layers 1
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer   
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%%
MiniBatchSize = 3;
valFrequency = floor(numel(imdsTrain.Files)/MiniBatchSize);
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer   
%     maxPooling2dLayer(2,'Stride',2)
    
    
     
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer 
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
%%
options1 = trainingOptions('sgdm', ...
    'MiniBatchSize',MiniBatchSize, ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',100, ...
    'Verbose',false, ...
    'Plots','training-progress');
%%
netTransfer1 = trainNetwork(imdsTrain,layers,options1);
YPred = classify(netTransfer1,imdsTest);
YTest = imdsTest.Labels;
netTransfer1BaselineAccuracy = sum(YPred == YTest)/numel(YTest)
%% Specify training options 
valFrequency = floor(numel(imdsTrain.Files)/MiniBatchSize);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',3, ...
    'MaxEpochs',35, ...
    'InitialLearnRate',.001, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',100, ...
    'LearnRateDropFactor',0.95, ... 
    'Verbose',false, ...
    'Plots','training-progress');
%% Train the network
net1 = trainNetwork(imdsTrain,layers,options);
analyzeNetwork(net1);
%% Report accuracy of baseline classifier on validation set
YPred = classify(net1,imdsTest);
YTest = imdsTest.Labels
imdsAccuracy = sum(YPred == YTest)/numel(YTest)

%% Plot confusion matrix
figure, plotconfusion(YTest,YPred)

%% PART 2: Baseline Classifier with Data Augmentation
%% Create augmented image data store



inputSize = [227 227 3];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXReflection',1,...
    'RandYReflection',1,...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);
%%
augimdsTrain = augmentedImageDatastore(imageSize,imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(imageSize,imdsTest);
valFrequency = floor(numel(augimdsTrain.Files)/MiniBatchSize);
options2 = trainingOptions('adam', ...
    'MiniBatchSize',MiniBatchSize, ...
    'MaxEpochs',35, ...
    'InitialLearnRate',.001, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',100, ...
    'LearnRateDropFactor',0.95, ... 
    'Verbose',false, ...
    'Plots','training-progress');

%% (OPTIONAL) Preview augmentation results 
batchedData = preview(augimdsTrain);
figure, imshow(imtile(batchedData.input))
    
%% Train the network. 
netTransfer = trainNetwork(augimdsTrain,layers,options2);
%Classify Validation Images
[YPred,scores] = classify(netTransfer,augimdsValidation);
%%
YValidation = imdsTest.Labels
augmented_accuracy = mean(YPred == YValidation)
%% Plot Confusion Matrix
figure, plotconfusion(YValidation,YPred)


