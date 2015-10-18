function [model] = clusterKmeans(X,K)
% [model] = clusterKmeans(X,K)
%
% K-means clustering

[N,D] = size(X);

% Choose random points to initialize means
means = zeros(K,D);
for k = 1:K
    i = ceil(rand*N);
    means(k,:) = X(i,:);
end

X2 = X.^2*ones(D,K);
iter = 1;
while 1
    means_old = means;
    
    % Compute Euclidean distance between each data point and each mean
    distances = sqrt(X2 + ones(N,D)*(means').^2 - 2*X*means');
    
    % Assign each data point to closest mean
    [~,clusters] = min(distances,[],2);
    
    % Compute mean of each cluster
    means = zeros(K,D);
    for k = 1:K
        means(k,:) = mean(X(clusters==k,:),1);
    end
   
    fprintf('Running K-means, difference = %f\n',max(max(abs(means-means_old))));
    
    if max(max(abs(means-means_old))) < 1e-5
        break;
    end
    iter = iter + 1;
end

model.means = means;
model.clusters = clusters;
model.predict = @predict;
end

function [clusters] = predict(model,X)
[N,D] = size(X);
means = model.means;
K = size(means,1);

distances = X.^2*ones(D,K) + ones(N,D)*(means').^2 - 2*X*means';
[~,clusters] = min(distances,[],2);
end
