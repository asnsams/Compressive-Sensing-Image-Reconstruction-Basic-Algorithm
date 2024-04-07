% Read Input
Orig_Image = im2double(imread('cameraman100.jpg'));
figure;
subplot(2,3,1);
imshow(Orig_Image)
title('Original Image');

% Compute DCT Matrix
F_Transform_M = dctmtx(size(Orig_Image,1));
Sparse_Image = F_Transform_M * Orig_Image * F_Transform_M';
subplot(2,3,2);
imshow(Sparse_Image)
title('Sparse Image');

% Get Sparse Vector
S = size(Orig_Image);
N = 1000;
Sparse_Vector = zeros(S(1)*S(2),1);
k_w = 1;  
for j = 1:S(2)
    for i = 1:S(1)
        Sparse_Vector(k_w,1) = Sparse_Image(i,j);
        k_w = k_w + 1;
    end
end

% Create a random Gaussian sampling matrix
num_samples = 6000;
Random_G_Sam_m = rand(num_samples,10*N) > 0.999;

Random_Sample_Y = Random_G_Sam_m * Sparse_Vector; 
theta = Random_G_Sam_m * dctmtx(size(10*N,1));

% Ensure CVX is setup correctly in your MATLAB environment for Basis Pursuit (L1 optimization)
cvx_begin
    variable x1(10*N);
    minimize( norm(x1, 1) );
    subject to
        theta * x1 == Random_Sample_Y;
cvx_end

% L2 optimization
cvx_begin
    variable x2(10*N);
    minimize( norm(x2, 2) );
    subject to
        theta * x2 == Random_Sample_Y;
cvx_end

% OMP algorithm for signal recovery
solution_omp = cs_omp(Random_Sample_Y, theta, 10*N)';

% Reshape and apply inverse DCT for L1 and L2 solutions
Solution_Pixel_Matrix_L1 = idct2(reshape(x1, [S(1), S(2)]));
Solution_Pixel_Matrix_L2 = idct2(reshape(x2, [S(1), S(2)]));

% Reshape OMP solution to matrix and apply inverse DCT
Solution_Pixel_Matrix_OMP = reshape(solution_omp, [S(1), S(2)]);
Solution_Pixel_Matrix_OMP = idct2(Solution_Pixel_Matrix_OMP);

% Plotting the reconstructed images
subplot(2,3,4);
imshow(Solution_Pixel_Matrix_L1);
title('Reconstructed by L1');

subplot(2,3,5);
imshow(Solution_Pixel_Matrix_L2);
title('Reconstructed by L2');

subplot(2,3,6);
imshow(Solution_Pixel_Matrix_OMP);
title('Reconstructed by OMP');

% Calculating and displaying reconstruction errors
error_L1 = norm(Solution_Pixel_Matrix_L1 - Orig_Image, 'fro') / norm(Orig_Image, 'fro');
error_L2 = norm(Solution_Pixel_Matrix_L2 - Orig_Image, 'fro') / norm(Orig_Image, 'fro');
error_OMP = norm(Solution_Pixel_Matrix_OMP - Orig_Image, 'fro') / norm(Orig_Image, 'fro');
disp(['Reconstruction error L1: ', num2str(error_L1)]);
disp(['Reconstruction error L2: ', num2str(error_L2)]);
disp(['Reconstruction error OMP: ', num2str(error_OMP)]);
