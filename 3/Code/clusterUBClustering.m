function [model] = clusterUBClustering(X,K,nModels)
% K clusters
% nModels times

[N,D] = size(X);

%% K-Means

for m = 1:nModels
    model.subModel{m} = clusterKmeans(X,K);
end

for m = 1:nModels
    % Each column of clustering is the prediction by the m model
    clusters(:,m) = model.subModel{m}.predict(model.subModel{m},X);
end

%% Naive Method

% The mode per row of clusters
% clusters = mode(clusters,2);
% model.clusters = clusters;

%% Ensemble Method

[R,runs] = size(clusters);

% Find similarity between each point with every other point
similarities = zeros(1,R);

for i = 1:R
    for j = 1:R
        rowInspect = clusters(i,:);
        rowCompare = clusters(j,:);
        similarity = sum(rowInspect == rowCompare) / runs;
        similarities(i,j) = similarity;
    end
end

% This will be the cluster of each object.
cluster = zeros(N,1);

% This variable will keep track of whether we've visited each object.
visited = zeros(N,1);

% K will count the number of clusters we've found
K = 0;

% eps will set the minimum percent of similar clustering
eps = 0.5;

% This variable will determine how many points must be neighbors to be clustered
minPoints = 1;

for i = 1:N
    if ~visited(i) 
        % We only need to consider examples that have never been visited
        visited(i) = 1;
        % Treat the similarities matrix like the "distance" matrix from DBSCAN
        % Find all Xi in column where similarity > 0.5
        neighbors = find(similarities(:,i) > eps);

        % If points exist, there is a new cluster
        if length(neighbors) >= minPoints
            K = K + 1;
            % Put i and each neighbor in the same cluster (or merge clusters if assigned)
            [visited,cluster] = expand(i,neighbors,K,eps,minPoints,similarities,visited,cluster);
        end
    end
end

model.clusters = cluster;
end

% i = i-th point being checked against other points
% neighbors = indices of points with D < eps
% K = cluster number
% eps = distance
% minPts = minimum points for cluster
% similarities = matrix of similarities between point cluster appearances
% visited = vector of objects that have been visited (1) or not (0)
% cluster = cluster of each object
function [visited,cluster] = expand(i,neighbors,K,eps,minPts,similarities,visited,cluster)
% Assign point i to cluster K
cluster(i) = K;
ind = 0;

while 1
    ind = ind + 1;
    
    if ind > length(neighbors)
        break;
    end
    
    % Get the neighboring point
    n = neighbors(ind);
    % Assign point to cluster K
    cluster(n) = K;
    
    if ~visited(n)
        % Set neighboring point to visited
        visited(n) = 1;
        
        % Treat the similarities matrix like the "distance" matrix from DBSCAN
        % Find all neighbors of this neighboring point in column where similarity >= 0.5
        neighbors2 = find(similarities(:,n) > eps);
        
        if length(neighbors2) >= minPts
            % This cluster is going to be expanded with new neighbors
            neighbors = [neighbors; setdiff(neighbors2,neighbors)];
        end
    end
end
end
