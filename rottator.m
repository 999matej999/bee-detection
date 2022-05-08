close all
clear
clc

%ds = dir('images\cropped\*.jpg');
ids = imageDatastore('images\cropped\', "IncludeSubfolders", true , "FileExtensions", ".jpg");

% img = read(ids);
% figure(1);
% imshow(img);
% reset(ids);
% 
% ids_rottated = flip(ids,1);
% 
% 
% img_rotated = read(ids_rottated);
% figure(2);
% imshow(img_rotated);

for img_i = 1:length(ids.Files)
    img = read(ids);
    filepath    = ids.Files(img_i); 
    filepath    = extractBefore(filepath, ".");
    filepath    = split(filepath, "\"); 
    filename    = filepath{end};
    foldername  = filepath{end-1};
    
    mkdir(['images\cropped_rottated\rottated_x\' foldername '\'])
    mkdir(['images\cropped_rottated\rottated_y\' foldername '\'])
    mkdir(['images\cropped_rottated\rottated_xy\' foldername '\'])

    for type = 1:3
        
        switch type
            case 1
                img_rotated = flip(img,2);
                imwrite(img_rotated, ['images\cropped_rottated\rottated_x\' foldername '\' filename '_x.jpg']);
            case 2
                img_rotated = flip(img,1);
                imwrite(img_rotated, ['images\cropped_rottated\rottated_y\' foldername '\' filename '_y.jpg']);
            case 3
                img_rotated = flip(img,2);
                img_rotated = flip(img_rotated,1);
                imwrite(img_rotated, ['images\cropped_rottated\rottated_xy\' foldername '\' filename '_xy.jpg']);
        end
    end
end
