load animals.mat

[n,d] = size(X);
X = standardizeCols(X);

%% Visualization
[U,S,V] = svd(X);

W = V(:,1:2)';
Z = X*W';

% Plotting
figure(1);
imagesc(Z);
figure(2);
scatter(Z(:,1),Z(:,2),'.');
gname(animals);