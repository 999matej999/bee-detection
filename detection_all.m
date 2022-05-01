positions = 12;
img_begin = [91; 173; 249; 327; 410; 489; 573; 661; 747; 831; 911; 988];
img_end = [141; 217; 299; 383; 470; 557; 641; 724; 804; 881; 956; 1028];
%img_avg_brightness = [129; 160; 172; 163; 123; 89; 83; 120; 160; 167; 159; 129];
img_avg_brightness = [188; 243; 245; 216; 161; 98; 96; 145; 206; 236; 231; 195];
img_thresholds = img_avg_brightness .* 0.1;

ds = dir('images/*.jpg');
ids = imageDatastore('images/*.jpg');

img = preview(ids);
reset(ids);
[im_h, im_w, ~] = size(img);

%avg_brightness = zeros(length(ds), 12);

for i = 1:length(ds)
    filename = ds(i).name
    img = read(ids);

    %imshow(rgb2gray(img));

    for j = 1:(positions)
        cropped_img = img(:,img_begin(j):img_end(j),:);
    
        %avg_brightness(i, j) = mean(cropped_img, 'all');
    
        if abs(avg_brightness(j) - mean(cropped_img, 'all')) > img_thresholds(j)
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
