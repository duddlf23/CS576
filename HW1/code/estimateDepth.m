function [depthMap, disparityMap] = estimateDepth(leftImage, rightImage, stereoParameters)
% This function estimate disparity and depth values from left and right
% images. You should calculate disparty map first and then convert the
% disparity map to depth map using left camera parameters.

% Function inputs:
% - 'leftImage': rectified left image.
% - 'rightImage': rectified right image.
% - 'stereoParameters': stereo camera parameters.

% Function outputs:
% - 'depth': depth map of left camera.
% - 'disparity': disparity map of left camera.

leftImageGray = rgb2gray(im2double(leftImage));
rightImageGray = rgb2gray(im2double(rightImage));

translation = stereoParameters.TranslationOfCamera2;
baseline = norm(translation);
focalLength = stereoParameters.CameraParameters1.FocalLength(1);


disparityMap = zeros(size(leftImageGray));
depthMap = zeros(size(leftImageGray));
% ----- Your code here (10) -----
[m, n] = size(leftImageGray);
max_disparity = 140;
w = 11;

gau_filter = fspecial('Gaussian', w, 1);
%leftImageGray = imfilter(leftImageGray, gau_filter, 'replicate');
%rightImageGray = imfilter(rightImageGray, gau_filter, 'replicate');

cost_vol = zeros(m, n, max_disparity);

pad_right = padarray(rightImageGray, [0 max_disparity], 'replicate', 'pre');

block_pad = round(w/2) - 1;

pad_left = padarray(leftImageGray,[block_pad block_pad], 'replicate', 'both');
padded_right = padarray(pad_right, [block_pad block_pad], 'replicate', 'both');


filter = zeros(w);
filter(1:w, 1:w) = 1 / (w^2);

avg_right = imfilter(pad_right, filter, 'replicate');
avg_left = imfilter(leftImageGray, filter, 'replicate');

for i=1:m
    for j=1:n
        for d = 1:max_disparity
            j2 = j -d + max_disparity;
            avg_l = avg_left(i, j);
            avg_r = avg_right(i,j2);
            A = pad_left(i:i+w-1, j:j+w-1);
            a = A(:) - avg_l;
            B = padded_right(i:i+w-1, j2:j2+w-1);
            b = B(:) - avg_r;
            cost_vol(i,j,d) = -dot(a,b) / (norm(a) * norm(b));
        end
    end
end
% for d = 1:max_disparity
%     cost_vol(:,:,d) = imfilter(cost_vol(:,:,d), gau_filter, 'replicate');
% end
% if m == 583
%     save('cost1.mat', 'cost_vol');
% else
%     save('cost2.mat', 'cost_vol');
% end
%[min_val, index] = min(cost_vol,[],3);

cost_vol2 = zeros(m, n, max_disparity);
parent = zeros(m, n, max_disparity);
min_dis = 10;
dis = min_dis:max_disparity;
dis_size = max_disparity - min_dis + 1;
% 

rob = @(x) x.^2 ./ (1 + x.^2);
cost_vol2(:,n,:) = cost_vol(:,n,:);
for i=1:m
    for j=n-1:-1:1
        for d = min_dis:max_disparity
            r = rob(d - dis);
            cost = reshape(cost_vol2(i,j + 1,min_dis:max_disparity), [dis_size, 1])  ...
                + reshape(r, [dis_size, 1]);
            [value, idx] = min(cost);
            cost_vol2(i,j,d) = cost_vol(i,j,d) + value;
            parent(i,j,d) = idx + min_dis - 1;
        end
    end
end
[~, idx] = min(cost_vol2(:,1,min_dis:max_disparity),[],3);
for i=1:m
    disparityMap(i,1) = idx(i) + min_dis - 1;
    idx_next = parent(i,1,idx(i) + min_dis - 1);
    for j=2:n
        disparityMap(i,j) = idx_next;
        idx_next = parent(i,j,idx_next);
    end
end
%[min_val, index] = min(cost_vol2(:,:,min_dis:max_disparity),[],3);
%disparityMap = index + min_dis - 1;
%disparityMap = imfilter(disparityMap, gau_filter, 'replicate');
depthMap = (focalLength * baseline) ./ disparityMap;