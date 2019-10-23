function [objective] = func_calibration(imagePoints, worldPoints, x)
% Objective function to minimize eq.10 in Zhang's paper. 
% Size of input variable x is 5+6*n where n is number of checkerboard 
% images. An intrinsic matrix can be reconstructed from first five
% parameters, and the extrinsic matrix can be reconstructed from remain
% parameters.

% You should fill the variable hat_m which contains reprojected positions 
% of checkerboard points in screen coordinate.

% Function inputs:
% - 'imagePoints': positions of checkerboard points in a screen space.
% - 'worldPoints': positions of checkerboard points in a model space.
% - 'x': parameters to be optimized.

% Function outputs:
% - 'objective': difference of estimated values and real values.
    
numView = size(imagePoints,3);
hat_m = zeros(size(imagePoints));

% ----- Your code here (9) -----

K = [1, 0, 0
     0, 1, 0
     0, 0, 1];
 
K(1,1) = x(1);
K(1,2) = x(2);
K(1,3) = x(3);
K(2,2) = x(4);
K(2,3) = x(5);

for view = 1:numView
    rvec = x(6*view+3:6*view+5);
    tvec = x(6*view:6*view+2);
    R = rotationVectorToMatrix(rvec);
    R = R';
    P = K * [R(:,1), R(:,2), reshape(tvec,[3,1])];
    points = worldPoints;
    points = [points, ones(size(imagePoints,1), 1)];
    %disp(size(points));
    hat = P * points';
    hat(1,:) = hat(1,:) ./ hat(3,:);
    hat(2,:) = hat(2,:) ./ hat(3,:);
    hat_m(:,:,view) = hat(1:2,:)';
end

objective = imagePoints - hat_m;
%objective = reshape(objective, [size(objective,1) * size(objective,2) * size(objective,3), 1]);
