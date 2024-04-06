% Main Script
clc; clear; close all;

% Read Input
Orig_Image = im2double(imread('cameraman.tif'));
figure;
subplot(1,3,1);
imshow(Orig_Image);
title('Original Image');

% Apply DCT to get Sparse Representation
DCT_Image = dct2(Orig_Image);
subplot(1,3,2);
imshow(log(abs(DCT_Image) + 1), []);
title('DCT of Image');

% Vectorize DCT representation
Sparse_Vector = DCT_Image(:);

% Create a random Gaussian sampling matrix
num_samples = round(length(Sparse_Vector) * 0.25); % For example, sample 25% of the entries
Random_G_Sam_m = randn(num_samples, length(Sparse_Vector));

% Sample the Sparse Vector
Random_Sample_Y = Random_G_Sam_m * Sparse_Vector; 

% Ask user for choice of reconstruction algorithm
disp('Choose Reconstruction Algorithm:');
disp('1: L1 Optimization');
disp('2: L2 Optimization');
disp('3: OMP');
disp('4: IHT');
disp('5: CoSaMP');
disp('6: SP');
disp('7: AMP');
choice = input('Enter your choice (1-7): ');

% Perform reconstruction based on user choice
switch choice
    case 1
        x_hat = L1Optimization(Random_Sample_Y, Random_G_Sam_m); % Note: pass measurement matrix, not DCT matrix
        algoName = 'L1 Optimization';
    case 2
        x_hat = L2Optimization(Random_Sample_Y, Random_G_Sam_m);
        algoName = 'L2 Optimization';
    case 3
        K = 20; % Example sparsity level
        x_hat = cs_omp(Random_Sample_Y, Random_G_Sam_m, K);
        algoName = 'OMP';
    case 4
        K = 20; maxIter = 200; % Example parameters
        x_hat = IHT(Random_Sample_Y, Random_G_Sam_m, K, maxIter);
        algoName = 'IHT';
    case 5
        K = 20; maxIter = 200; % Example parameters
        x_hat = CoSaMP(Random_Sample_Y, Random_G_Sam_m, K, maxIter);
        algoName = 'CoSaMP';
    case 6
        K = 20; maxIter = 200; % Example parameters
        x_hat = SP(Random_Sample_Y, Random_G_Sam_m, K, maxIter);
        algoName = 'SP';
    case 7
        maxIter = 200; % Example parameters
        lambda = 0.1; % Assumed threshold for AMP (adapt as needed)
        x_hat = AMP(Random_Sample_Y, Random_G_Sam_m, maxIter); % Adapt AMP to use lambda if needed
        algoName = 'AMP';
    otherwise
        disp('Invalid choice.');
        return;
end

% Reshape and apply inverse DCT for the solution
Solution_DCT_Matrix = reshape(x_hat, size(DCT_Image));
Solution_Pixel_Matrix = idct2(Solution_DCT_Matrix);

% Plotting the reconstructed image
subplot(1,3,3);
imshow(Solution_Pixel_Matrix);
title(['Reconstructed by ', algoName]);



function x_hat_L1 = L1Optimization(y, A)
% L1Optimization Reconstructs a sparse signal using L1 optimization (Basis Pursuit)
%   y: Measurement vector
%   A: Measurement matrix

cvx_begin quiet
    variable x_hat_L1(size(A,2));
    minimize( norm(x_hat_L1, 1) );
    subject to
        A * x_hat_L1 == y;
cvx_end
end

function x_hat_L2 = L2Optimization(y, A)
% L2Optimization Reconstructs a signal using L2 optimization
%   y: Measurement vector
%   A: Measurement matrix

cvx_begin quiet
    variable x_hat_L2(size(A,2));
    minimize( norm(x_hat_L2, 2) );
    subject to
        A * x_hat_L2 == y;
cvx_end
end

function x_hat_OMP = cs_omp(y, A, K)
% cs_omp Reconstructs a sparse signal using Orthogonal Matching Pursuit
%   y: Measurement vector
%   A: Measurement matrix
%   K: Sparsity level (number of non-zero elements in the signal)

N = size(A,2);
x_hat_OMP = zeros(N,1);
residual = y;
support_set = false(N,1);

for iter = 1:K
    [~, idx] = max(abs(A' * residual));
    support_set(idx) = true;
    x_temp = zeros(N,1);
    x_temp(support_set) = A(:,support_set) \ y;
    residual = y - A * x_temp;
end

x_hat_OMP(support_set) = A(:,support_set) \ y;
end

function x_hat = IHT(y, A, K, maxIter)
% IHT Reconstructs a sparse signal using Iterative Hard Thresholding
%   y: Measurement vector
%   A: Measurement matrix
%   K: Sparsity level
%   maxIter: Maximum number of iterations

x_hat = zeros(size(A,2), 1); % Initial estimate
for iter = 1:maxIter
    x_hat = x_hat + A' * (y - A * x_hat); % Gradient descent step
    [~, idx] = sort(abs(x_hat), 'descend'); % Sort by magnitude
    x_hat(idx(K+1:end)) = 0; % Hard thresholding
end
end


function x_hat = CoSaMP(y, A, K, maxIter)
% CoSaMP Reconstructs a sparse signal using CoSaMP algorithm
%   y: Measurement vector
%   A: Measurement matrix
%   K: Sparsity level
%   maxIter: Maximum number of iterations

x_hat = zeros(size(A,2), 1);
for iter = 1:maxIter
    r = y - A * x_hat;
    omega = sort(abs(A' * r), 'descend');
    T = union(find(x_hat), omega(1:2*K));
    b = zeros(size(A,2), 1);
    b(T) = pinv(A(:,T)) * y;
    [~, idx] = sort(abs(b), 'descend');
    x_hat = zeros(size(A,2), 1);
    x_hat(idx(1:K)) = b(idx(1:K));
end
end


function x_hat = SP(y, A, K, maxIter)
% SP Reconstructs a sparse signal using Subspace Pursuit
%   y: Measurement vector
%   A: Measurement matrix
%   K: Sparsity level
%   maxIter: Maximum number of iterations

x_hat = zeros(size(A,2), 1);
T = sort(abs(A' * y), 'descend');
T = T(1:K);
for iter = 1:maxIter
    x_temp = pinv(A(:,T)) * y;
    [~ , idx] = sort(abs(x_temp), 'descend');
    T = T(idx(1:K));
    x_hat = zeros(size(A,2), 1);
    x_hat(T) = x_temp(idx(1:K));
    if norm(y - A*x_hat) < 1e-6
        break;
    end
end
end

function x_hat = AMP(y, A, maxIter)
% AMP Reconstructs a sparse signal using Approximate Message Passing
%   y: Measurement vector
%   A: Measurement matrix
%   maxIter: Maximum number of iterations

[m, n] = size(A);
x_hat = zeros(n, 1);
z = y;

for iter = 1:maxIter
    x_prev = x_hat;
    x_hat = x_hat + A' * z;
    x_hat = softThreshold(x_hat, lambda); % Implement softThreshold based on your thresholding strategy
    z = y - A * x_hat + (1/m) * z * sum(x_hat ~= x_prev);
end
end

function x = softThreshold(x, lambda)
% Soft thresholding function for AMP
x = sign(x) .* max(abs(x) - lambda, 0);
end


