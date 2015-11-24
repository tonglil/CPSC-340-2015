%% Load data
load YaleB_32x32.mat
X = fea;
[n,d] = size(X);

% Places the faces in a random order
perm = randperm(n);
X = X(perm,:);

% Show examples of faces
figure(1);title('Original faces');
for i = 1:16
    subplot(4,4,i);
    imagesc(reshape(X(i,:),32,32));
    colormap('gray');
end

%% Run PCA
k = 100;
model = dimRedSPCA(X,k);

% Make low-dimensional representation
Z = model.compress(model,X);

% Look at low-dimensional representation in original space
Xhat = model.expand(model,Z);

%% Visualize results

% Show mean image
figure(2);
imagesc(reshape(model.mu,32,32));
colormap('gray');
title('Mean Face');

% Show Eigenfaces
figure(3);clf;
for i = 1:k
    subplot(ceil(sqrt(k)),ceil(sqrt(k)),i);
    imagesc(reshape(model.W(i,:),32,32));
    colormap('gray');
end

% Show reconstruction of faces from figure(1)
figure(4);
subplot(1,2,1);
imagesc(X);
title('Original Data');
subplot(1,2,2);
imagesc(Z)
title(sprintf('Data compressed to %d numbers',k));

% Show reconstruction of faces from figure(1)
figure(5);
for i = 1:16
    subplot(4,4,i);
    imagesc(reshape(Xhat(i,:),32,32));
    colormap('gray');
end