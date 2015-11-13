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

%% Compression
for k = 1:3
    W = V(:,1:k)';
    Z = X*W'; % Z is n-by-k

    ZW = Z*W;
    compressionRatio = norm(X - ZW,'fro')^2 / norm(X,'fro')^2
end

singularValues = diag(S(:,1:n));

% Compression ratio for k
cr = 1 - cumsum(singularValues.^2 ./ sum(singularValues.^2))