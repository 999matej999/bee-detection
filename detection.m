img = imread('images/Image0314 11-40-24.jpg');
imshow(rgb2gray(img));
summ = sum(rgb2gray(255-img), 1);
%findpeaks(summ);
%findpeaks(summ, 'MinPeakDistance', 50);
[peakValues, indexes] = findpeaks(summ, 'MinPeakDistance', 50);
%plot(indexes)

%avg_brightness = zeros(numel(indexes)-1, 1);

avg_brightness = [129; 160; 172; 163; 123; 89; 83; 120; 160; 167; 159; 129];

for i = 1:(numel(indexes)-1)
    cropped_img = img(:,indexes(i):indexes(i+1),:);

    %avg_brightness(i) = mean(cropped_img, 'all');

    if abs(avg_brightness(i) - mean(cropped_img, 'all')) > 10
        testImage = cropped_img;
        %testLabel = testSet.Labels(1)
        
        % Create augmentedImageDatastore to automatically resize the image when
        % image features are extracted using activations.
        ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');
        
        % Extract image features using the CNN
        imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');
        
        % Make a prediction using the classifier
        predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
        disp([num2str(i), ': ', char(predictedLabel)]);
    end

end
