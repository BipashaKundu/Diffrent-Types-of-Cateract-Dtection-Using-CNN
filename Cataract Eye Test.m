clc;
clear all;
Ratio = [227 227];
load('net.mat');

I=imread("10.jpg");
J=cropping(I);
B=medfilt3(J, [5 5 5]);
[BW,rgb] = createMask(J);
BW1 = imfill(BW,'holes');
figure,
imshow(BW1);
class=bwareaopen(BW1,1250);
numberOfTruePixels = sum(class(:));
B =(numberOfTruePixels>=1000);

if B(1,1)==1
C=imread("3.jpg");
I_input=imresize(C,[227 227]);
cat = imresize(I_input,Ratio); 
imwrite(cat,"cat.png");
    
imds = imageDatastore("cat.png");
Pred_R = classify(net,imds);
    
%% output display
figure;
subplot(1,2,1)
imshow(I_input);
title("Input Image");
subplot(1,2,2);
imshow(I_input);
title( "Cateract: " + string(Pred_R));
else
f2 = msgbox('Cataract Healthy eye');
end

