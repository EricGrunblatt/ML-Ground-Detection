%Bayesian Ground Detection Assignment

%Housekeeping
clc;
close all;
warning('off', 'Images:initSize:adjustingMag');

%File directory
datadir = 'Greenhouse\\Tunnel-';

trainingImages = [1,2,3];
testingImages = (3);

%Parameters
nDim = 256; %number of bins for the color likelihood distribution. This is too big. Try to have smaller bins such as 8, 16, 32, etc.

%Training process
Pr_x_given_y_equalsTo_1 = zeros(nDim,nDim,nDim); %likelihood for the ground class
Pr_x_given_y_equalsTo_0 = zeros(nDim,nDim,nDim); %likelihood for the non-ground class
N_GroundPixels = 0; %Pr_y_equalsTo_1 = N_GroundPixels/N_totalPixels
N_TotalPixels = 0;
for iFile = trainingImages
    %Load the training image and labeled image regions
    origIm = imread(sprintf('%s%02d.jpg',datadir,iFile));    
    labels = imread(sprintf('%s%02d-label.png',datadir,iFile)); %label=1 representing the ground class

    %Visualization input image and its labels
    [nrows,ncols,~]= size(origIm);
    showIm = origIm; showIm(labels==1) = 255;
    figure; imshow([origIm repmat(255*labels, [1 1 3]) showIm],[]); title('Training image, GT, and GT overlaind on the training image');

    %Prior-related codes:
    for i = 1:nrows
        for j = 1:ncols
            if(labels(i,j) == 1)
                N_GroundPixels = N_GroundPixels + 1;
            end
        end
    end
    N_TotalPixels = N_TotalPixels + (nrows * ncols);
    
    %Likelihood-related codes:
    %Looping through each pixel in each row
    for i = 1:nrows
        for j = 1:ncols
            r = origIm(i,j,1) + 1;
            g = origIm(i,j,2) + 1;
            b = origIm(i,j,3) + 1;
            %if label(i,j) == 1, then it is a ground pixel, otherwise it is not
            if(labels(i,j) == 1)
                Pr_x_given_y_equalsTo_1(r,g,b) = Pr_x_given_y_equalsTo_1(r,g,b) + 1;
            else
                Pr_x_given_y_equalsTo_0(r,g,b) = Pr_x_given_y_equalsTo_0(r,g,b) + 1;
            end      
        end
    end
end

%Some other codes such as normalizing the likelihood/prior and computing Pr_y_equalsTo_0:
Pr_x_given_y_equalsTo_1 = Pr_x_given_y_equalsTo_1 / N_TotalPixels;
Pr_x_given_y_equalsTo_0 = Pr_x_given_y_equalsTo_0 / N_TotalPixels;
Pr_y_equalsTo_1 = N_GroundPixels / (N_TotalPixels);
Pr_y_equalsTo_0 = 1 - Pr_y_equalsTo_1;

%Testing
truePositives = 0;
falsePositives = 0;
falseNegatives = 0;
macroFScore = 0;
for iFile = testingImages  
    %Load the testing image and ground truth regions  
    origIm = imread(sprintf('%s%02d.jpg',datadir,iFile));    
    gtMask = imread(sprintf('%s%02d-label.png',datadir,iFile)); 
    [nrows, ncols,~] = size(origIm);  
    
    %Define the posterior
    Pr_y_equalsTo_1_given_x = zeros(nrows,ncols);   
    Pr_y_equalsTo_0_given_x = zeros(nrows,ncols);
    
    %Codes to obtain the final classification result (detectedMask):
    detectedMask = zeros(nrows, ncols);
    for i = 1:nrows
        for j = 1:ncols
            r = origIm(i,j,1) + 1;
            g = origIm(i,j,2) + 1;
            b = origIm(i,j,3) + 1;
            %Calculates the posterior
            Pr_y_equalsTo_1_given_x(i,j) = Pr_x_given_y_equalsTo_1(r,g,b) * Pr_y_equalsTo_1;
            Pr_y_equalsTo_0_given_x(i,j) = Pr_x_given_y_equalsTo_0(r,g,b) * Pr_y_equalsTo_0;
            if(Pr_y_equalsTo_1_given_x(i,j) > Pr_y_equalsTo_0_given_x(i,j))
                detectedMask(i,j) = detectedMask(i,j) + 1;
            end
        end
    end
    
    %Codes to calculate the TP, FP, FN:
    tempTP = 0;
    tempFP = 0;
    tempFN = 0;
    for i = 1:nrows
        for j = 1:ncols
            if(gtMask(i,j) == 1 && detectedMask(i,j) == 1) %True Positive: Label and original image pixels identify as ground pixels
                truePositives = truePositives + 1;
                tempTP = tempTP + 1;
            elseif(gtMask(i,j) == 0 && detectedMask(i,j) == 1) %False Positive: Label pixel does not identify as ground pixel, but original image pixel does
                falsePositives = falsePositives + 1;
                tempFP = tempFP + 1;
            elseif(gtMask(i,j) == 1 && detectedMask(i,j) == 0) %False Negative: Label pixel identifies as ground pixel, but original image pixel does not
                falseNegatives = falseNegatives + 1;
                tempFN = tempFN + 1;
            end
        end
    end
    
    %Calculates precision, recall, and f-score for image alone, adds to macroFScore
    tempPrecision = tempTP / (tempTP + tempFP);
    tempRecall = tempTP / (tempTP + tempFN);
    tempFScore = (2 * tempPrecision * tempRecall) / (tempPrecision + tempRecall);
    macroFScore = macroFScore + tempFScore;
    
    %Visualize the classification results
    showIm = origIm; showIm(detectedMask==1) = 255;
    figure; imshow([origIm repmat(255*detectedMask,[1 1 3]) showIm],[]);
end

%Codes to calculate the precision, recall, and fscore:
%Precision
precision = truePositives / (truePositives + falsePositives);
%Recall
recall = truePositives / (truePositives + falseNegatives);
%Micro F-Score
microFScore = (2 * precision * recall) / (precision + recall);
%Macro F-Score
macroFScore = macroFScore / length(testingImages);

fprintf('Precision: %f\n',precision);
fprintf('Recall: %f\n',recall);
fprintf('Micro F-Score: %f\n',microFScore);
fprintf('Macro F-Score: %f\n',macroFScore);
