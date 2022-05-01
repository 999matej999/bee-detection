avg_brightness = [129; 160; 172; 163; 123; 89; 83; 120; 160; 167; 159; 129];

ds = dir('images/*.jpg');
ids = imageDatastore('images/*.jpg');

img = preview(ids);
reset(ids);
[im_h, im_w, ~] = size(img);

for i = 1:length(ds)
    filename = ds(i).name
    img = read(ids);

    imshow(rgb2gray(img));
    summ = sum(rgb2gray(255-img), 1);
    [peakValues, indexes] = findpeaks(summ, 'MinPeakDistance', 50);

    for j = 1:(numel(indexes)-1)
        cropped_img = img(:,indexes(j):indexes(j+1),:);
    
        %avg_brightness(i) = mean(cropped_img, 'all');
    
        if abs(avg_brightness(j) - mean(cropped_img, 'all')) > 10
            testImage = cropped_img;
            %testLabel = testSet.Labels(1)
            
            % Create augmentedImageDatastore to automatically resize the image when
            % image features are extracted using activations.
            ds_ = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');
            
            % Extract image features using the CNN
            imageFeatures = activations(net, ds_, featureLayer, 'OutputAs', 'columns');
            
            % Make a prediction using the classifier
            predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
            disp([num2str(j), ': ', char(predictedLabel)]);
        end
    
    end
end
