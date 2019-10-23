function [cameraParams] = estimateSingleCameraParameters(imagePoints, boardSize, patchSize, imageSize)
% This function will estimate camera parameters (intrinsic, extrinsic) from
% checkerboard image points.

% Zhang's method consists of 5 parts
% 1. Estimate homography from checkerboard plane to screen space.
% 2. Calculate B matrix by solving Vb = 0.
% 3. Extract intrinsic parameters from B matrix.
% 4. Calculate extrinsic parameters from intrinsic parameters and homography.
% 5. Refine parameters using the maximum likelihood estimation.

% Function inputs:
% - 'imagePoints': positions of checkerboard points in a screen space.
% - 'boardSize': the number of horizontal, vertical patchs in the checkerboard.
% - 'patchSize': the size of the checkerboard patch in mm.
% - 'imageSize': the size of the checkerboard image in pixels.

% Function outputs:
% - 'cameraParams': a camera parameter includes intrinsic and extrinsic.

numView = size(imagePoints, 3);
numVerticalPatch = boardSize(1) - 1;
numHorizontalPatch = boardSize(2) - 1;
numCorner = size(imagePoints, 1);

%% Estimate a homography (appendix A)
% Generate checkerboard world points
worldPoints = zeros(size(imagePoints,1), size(imagePoints,2));

% Fill worldPoints (positions of checkerboard corners)
% ----- Your code here (1) ----- (slide 6)
y = 100;
idx = 0;
for j = 1:numHorizontalPatch
    x = 250;
    for i = 1:numVerticalPatch
        idx = idx + 1;
        worldPoints(idx, :) = [x, y];
        x = x - patchSize;
    end
    y = y + patchSize;
end
%disp(worldPoints);
%disp(size(imagePoints));


% Build L matrix
L = zeros(2 * numCorner, 9, numView);

% Fill L matrix
% ----- Your code here (2) ----- (slide 13)
for k  = 1:numView
    for i = 1:numCorner
        X = worldPoints(i,1);
        Y = worldPoints(i,2);
        u = imagePoints(i,1,k);
        v = imagePoints(i,2,k);
        L(i*2-1,:,k) = [-X, -Y, -1, 0, 0, 0, u*X, u*Y, u];
        L(i*2,:,k) = [0, 0, 0, -X, -Y, -1, v*X, v*Y, v];
    end
end
    

% Calculate a homography using SVD
homography = zeros(3,3,numView);

% Fill homography matrix
% ----- Your code here (3) ----- (slide 15)
for k = 1:numView
    [~, ~, V] = svd(L(:,:,k));
    h = V(:,9);
    h = h / h(9);
    homography(:,:,k) = reshape(h, [3,3])';
end
%disp(homography);

%% Solve closed-form (section 3.1)
V = zeros(2 * numView, 6);
b = zeros(6, 1);

% Fill V matrix and calculate b vector
% ----- Your code here (4) ----- (slide 19, 23)
for view = 1:numView
    h = homography(:,:,view);
    k = 1;
    l = 2;
    V(2*view - 1,:) = [h(1,k)*h(1,l), h(1,k)*h(2,l) + h(2,k)*h(1,l), ...
        h(1,k)*h(3,l) + h(3,k)*h(1,l), h(2,k)*h(2,l), h(2,k)*h(3,l) + ...
        h(3,k)*h(2,l), h(3,k)*h(3,l)];
    l = 1;
    V(2*view,:) = [h(1,k)*h(1,l), h(1,k)*h(2,l) + h(2,k)*h(1,l), ...
        h(1,k)*h(3,l) + h(3,k)*h(1,l), h(2,k)*h(2,l), h(2,k)*h(3,l) + ...
        h(3,k)*h(2,l), h(3,k)*h(3,l)];
    k = 2;
    l = 2;
    V(2*view,:) = V(2*view,:) - [h(1,k)*h(1,l), h(1,k)*h(2,l) + h(2,k)*h(1,l), ...
        h(1,k)*h(3,l) + h(3,k)*h(1,l), h(2,k)*h(2,l), h(2,k)*h(3,l) + ...
        h(3,k)*h(2,l), h(3,k)*h(3,l)];
    
end

[~, ~, V_hat] = svd(V);

B = V_hat(:, 6);

%disp(B);

%% Extraction of the intrinsic parameters from matrix B (appendix B)

% ----- Your code here (5) ----- (slide 24)
v0 = (B(2)*B(3) - B(1)*B(5)) / (B(1)*B(4) - B(2)^2);  % modify this line
lambda = B(6) - (B(3)^2 + v0 * (B(2)*B(3) - B(1)*B(5))) / B(1);  % modify this line
alpha = (lambda / B(1))^0.5;  % modify this line
beta = (lambda*B(1) / (B(1)*B(4) - B(2)^2))^0.5;  % modify this line
gamma = -B(2)*alpha^2*beta/lambda;  % modify this line
u0 = gamma*v0/beta - B(3)*alpha^2/lambda;  % modify this line



%% Estimate initial RT (section 3.1)
Rt = zeros(3, 4, numView);
k = zeros(3,3);
k(1,1) = alpha;
k(1,2)= gamma;
k(1,3) = u0;
k(2,2) = beta;
k(2,3) = v0;
k(3,3) = 1;
k_inv = inv(k);
% Fill Rt matrix
% ----- Your code here (6) ----- (slide 25, 26)
for view = 1:numView
    h1 = homography(:,1,view);
    h2 = homography(:,2,view);
    h3 = homography(:,3,view);
    lambda_prime = (1 / norm(k_inv*h1) + 1 / norm(k_inv*h2)) / 2;
    Rt(:,1,view) = lambda_prime * k_inv * h1;
    Rt(:,2,view) = lambda_prime * k_inv * h2;
    Rt(:,3,view) = cross(Rt(:,1,view), Rt(:,2,view));
    Rt(:,4,view) = lambda_prime * k_inv * h3;
    [U, ~, V] = svd(Rt(:,1:3,view));
    Rt(:,1:3,view) = U*V';
end

%disp(Rt);


%% Maximum likelihood estimation (section 3.2)
options = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt', ...
    'TolX', 1e-32, 'TolFun', 1e-32, 'MaxFunEvals', 1e64, ...
    'MaxIter', 1e64, 'UseParallel', true);

% options = optimoptions(@lsqnonlin, 'Algorithm', 'trust-region-reflective', ...
%     'TolX', 1e-32, 'TolFun', 1e-32, 'MaxFunEvals', 1e64, ...
%     'MaxIter', 2, 'UseParallel', true);
% options = optimoptions(@lsqnonlin, 'Algorithm', 'trust-region-reflective', ...
%     'TolX', 1e-32, 'TolFun', 1e-32, 'MaxFunEvals', 1e64, ...
%     'MaxIter', 2, 'UseParallel', true);
% Build initial x value as x0
% ----- Your code here (7) ----- (slide 29)

% 5 for intrinsic
% 3 for translation, 3 for rotation, total 6 for each checkerboard image
x0 = zeros(5 + 6 * size(imagePoints, 3), 1);  % modify this line
x0(1) = alpha;
x0(2) = gamma;
x0(3) = u0;
x0(4) = beta;
x0(5) = v0;
for view = 1:numView
    x0(6*view:6*view+2) = Rt(:,4,view);
    R = Rt(:,1:3,view);
    R = R';
    rvec = rotationMatrixToVector(R);
    x0(6*view+3:6*view+5) = rvec;
end

% Non-least square optimization
% Read [https://mathworks.com/help/optim/ug/lsqnonlin.html] for more information
[objective] = @(x) func_calibration(imagePoints, worldPoints, x);
[x_hat, ~, ~, ~, ~] = lsqnonlin(objective,x0,[], [],options);

%disp(size(x_hat));
%lb
%ub
%x_hat = x0;
%% Build camera parameters
rvecs = zeros(numView, 3);
tvecs = zeros(numView, 3);
K = [1, 0, 0
     0, 1, 0
     0, 0, 1];

% Extract intrinsic matrix K, rotation vectors and translation vectors from x_hat
% ----- Your code here (8) -----
K(1,1) = x_hat(1);
K(1,2) = x_hat(2);
K(1,3) = x_hat(3);
K(2,2) = x_hat(4);
K(2,3) = x_hat(5);
for view = 1:numView
    rvecs(view,:) = x_hat(6*view+3:6*view+5);
    tvecs(view,:) = x_hat(6*view:6*view+2);
end


% Generate cameraParameters structure
cameraParams = cameraParameters('IntrinsicMatrix', K', ...
    'RotationVectors', rvecs, 'TranslationVectors', tvecs, ...
    'WorldPoints', worldPoints, 'WorldUnits', 'mm', ...
    'imageSize', imageSize) ; 


reprojected_errors = zeros(size(imagePoints));

% Uncomment this line after you implement this function to calculate
% reprojection errors of your camera parameters.
 reprojected_errors = imagePoints - cameraParams.ReprojectedPoints;

cameraParams = cameraParameters('IntrinsicMatrix', K', ...
    'RotationVectors', rvecs, 'TranslationVectors', tvecs, ...
    'WorldPoints', worldPoints, 'WorldUnits', 'mm', ...
    'imageSize', imageSize, 'ReprojectionErrors', reprojected_errors) ; 