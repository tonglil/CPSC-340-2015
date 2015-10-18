function [model] = knn(X,y,K)
% [model] = knn(X,y,k)
%
% Implementation of k-nearest neighbour classifer

model.X = X;        % coordinates of point
model.y = y;        % labels for X
model.K = K;        % K in KNN
model.C = max(y);   % number of objects
model.predict = @predict;
end

function [yhat] = predict(model,Xtest)
X = model.X;
K = model.K;

[N,D] = size(X);        % [rows,columns] of X [250,2]
[T,D] = size(Xtest);    % [rows,columns] of Xtest [250,2]
Distances = X.^2 * ones(D,T) + ones(N,D) * (Xtest').^2 - 2 * X * (Xtest');  % each row is the distances of Xi from Xtesti

yhat = zeros(T,1);

% for each row of distances, compute the yHat
for i = 1:T
    [~, sIndex] = sort(Distances(:,i));    % sort a row of distances from small to large
    kIndex = sIndex(1:K);      % get k indexes from sorted indexes
    
    neighbors = model.y(kIndex);    % select the value of the neighbors from y using the indexes
    
    yhat(i) = mode(neighbors);      % pick the most common neighbor
end
end