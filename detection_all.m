positions = 12;
img_begin = [91; 173; 249; 327; 410; 489; 573; 661; 747; 831; 911; 988];
img_end = [141; 217; 299; 383; 470; 557; 641; 724; 804; 881; 956; 1028];
img_avg_brightness = [188; 243; 245; 216; 161; 98; 96; 145; 206; 236; 231; 195];
img_thresholds = img_avg_brightness .* 0.1;

ds = dir('images/*.jpg');
ids = imageDatastore('images/*.jpg');

img = preview(ids);
reset(ids);
[im_h, im_w, ~] = size(img);

bee_in = 0;
bee_out = 0;
bees = 0;

for i = 1:length(ds)
    filename = ds(i).name
    img = read(ids);

    %imshow(rgb2gray(img));

    for j = 1:(positions)
        cropped_img = img(:,img_begin(j):img_end(j),:);
    
    
        if abs(img_avg_brightness(j) - mean(cropped_img, 'all')) > img_thresholds(j)
            testImage = cropped_img;
            
            % Create augmentedImageDatastore to automatically resize the image when
            % image features are extracted using activations.
            ds_ = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');
            
            % Extract image features using the CNN
            imageFeatures = activations(net, ds_, featureLayer, 'OutputAs', 'columns');
            
            % Make a prediction using the classifier
            predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
            disp([num2str(j), ': ', char(predictedLabel)]);

            if (predictedLabel == "bee_complete_in")
                bee_in = bee_in + 1;
            elseif (predictedLabel == "bee_complete_out")
                bee_out = bee_out + 1;
            end
            bees = bee_in - bee_out
        end
    
    end
end
