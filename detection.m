img = imread('images/Image0538 11-41-09.jpg');
imshow(rgb2gray(img));

positions = 12;
img_begin = [91; 173; 249; 327; 410; 489; 573; 661; 747; 831; 911; 988;];
img_end = [141; 217; 299; 383; 470; 557; 641; 724; 804; 881; 956; 1028;];

avg_brightness = zeros(numel(12, 1));

%avg_brightness = [129; 160; 172; 163; 123; 89; 83; 120; 160; 167; 159; 129];

for i = 1:(positions)
    cropped_img = img(:,img_begin(i):img_end(i),:);

    avg_brightness(i) = mean(cropped_img, 'all');

    if false%abs(avg_brightness(i) - mean(cropped_img, 'all')) > 10
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
