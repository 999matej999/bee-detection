close all
clear
clc

%ds = datastore('images\*.jpg')
ds = dir('images\*.jpg');
ids = imageDatastore('images\*.jpg');

img = preview(ids);
reset(ids);
[im_h, im_w, ~] = size(img)

% ts = tabularTextDatastore('annotations',"TextscanFormats","%s");

%     count = numel(ds.Files);
%     known_result = extractAfter(ds.Files, "images");
%     filename = extractAfter(known_result, "\");
%     known_result = extractBefore(filename, ".");
%     annotation_file = fullfile([known_result '.txt'])



for i = 1:length(ds)
    %known_result = extractAfter(ds(files).name), "images");
    filename = ds(i).name
    known_result = extractBefore(filename, ".")
    annotation_filename = ['annotations\' known_result '.txt'];
    f = fopen(annotation_filename); 
    if(f > 0)
        Yolo = textscan(f,'%d %f %f %f %f');
        s = size(Yolo{1});
        img = read(ids);
        %imshow(img);
        %hold on;
        for annot = 1:s(1)
            class = Yolo{1}(annot);
            x = Yolo{2}(annot) * im_w;
            y = Yolo{3}(annot) * im_h;
            w = Yolo{4}(annot) * im_w;
            h = Yolo{5}(annot) * im_h;
            left    = uint16((x - w / 2));
            right   = uint16((x + w / 2));
            top     = uint16((y - h / 2));
            bottom  = uint16((y + h / 2));
            img_crop = imcrop(img, [left, top, w, h]);
            %imshow(img_crop);
            annot_s = fullfile(string(annot));
            %rectangle('Position', [left, top, w, h], 'EdgeColor', 'r'); %// draw rectangle on image
            switch class
                case 0
                    imwrite(img_crop, ['images\bee_complete\' known_result '_' num2str(annot) '.jpg']);
                    %imwrite(img, ['images\bee_complete\' known_result '.jpg']);
                case 1
                    imwrite(img_crop, ['images\bee_head\' known_result '_' num2str(annot) '.jpg']);
                    %imwrite(img, ['images\bee_head\' known_result '.jpg']);
                case 2
                    imwrite(img_crop, ['images\bee_abdomen\' known_result '_' num2str(annot) '.jpg']);
                    %imwrite(img, ['images\bee_abdomen\' known_result '.jpg']);
                case 3
                    imwrite(img_crop, ['images\bee_cluster\' known_result '_' num2str(annot) '.jpg']);
                    %imwrite(img, ['images\bee_cluster\\' known_result '.jpg']);
            end
        end

        fclose(f);
    else
        img = read(ids);
    end
        
end