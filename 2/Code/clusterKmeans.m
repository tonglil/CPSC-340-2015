function [model] = clusterKmeans(X,K)
% [model] = clusterKmeans(X,K)
%
% K-means clustering

[N,D] = size(X);

X2 = X.^2*ones(D,K);

means = zeros(K,D);

% Choose first point randomly
i = ceil(rand*N);
means(1,:) = X(i,:);

% Determine the other points
for k = 2:K
    % Compute the distance of every point to the closest mean
    meanDistances = sqrt(X2 + ones(N,D)*(means').^2 - 2*X*means');
    
    % Select the minimum distance (that was calculated to an initialized point) of each row
    minDistances = min(meanDistances(:,1:k), [], 2);
    
    % Square the distance to emphasize porportionality
    minDistances2 = minDistances.^2;
    
    % Normalize the distances
    normalizedMeanD = minDistances2 / sum(minDistances2);
    
    % Sample the next point proportional to these distances squared
    i = sampleDiscrete(normalizedMeanD);
    
    means(k,:) = X(i,:);
end

% means       = zeros(K, D);
% minEucDisSQ = ones(N, 1);
% 
% for k = 1:K
%     % Create sampling distribution for N,1
%     P = bsxfun(@rdivide, minEucDisSQ, sum(minEucDisSQ));
% 
%     % Choose a new mean candidate from P for K,D
%     means(k,:) = X(sampleDiscrete(P), :);
% 
%     % Compute ED for N,K
%     EucDis = sqrt(X.^2*ones(D,K) + ones(N,D)*(means').^2 - 2*X*means');
% 
%     % Select Min ED^2 for N,1
%     minEucDisSQ = bsxfun(@power, min(EucDis(:,1:k), [], 2), 2*ones(N, 1));
% end

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
    
    % If we only have two features, make a colored scatterplot
    if D == 2
        clf;hold on;
        colors = getColors;
        for k = 1:K
            h = plot(X(clusters==k,1),X(clusters==k,2),'.');
            set(h,'Color',colors{k});
        end
        pause(.25);
    end
    
    fprintf('Running K-means, difference = %f\n',max(max(abs(means-means_old))));
    
    if max(max(abs(means-means_old))) < 1e-5
        break;
    end
end

model.means = means;
model.clusters = clusters;
