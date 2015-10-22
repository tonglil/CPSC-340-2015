function [model] = clusterDBcluster(X,eps,minPts)

[N,D] = size(X);
% N = rows (objects)
% D = columns (features)

% Compute distances between all points
D = X.^2*ones(D,N) + ones(N,D)*(X').^2 - 2*X*X';

% This will be the cluster of each object.
cluster = zeros(N,1);

% This variable will keep track of whether we've visited each object.
visited = zeros(N,1);

% K will count the number of clusters we've found
K = 0;
for i = 1:N
    if ~visited(i) 
        % We only need to consider examples that have never been visited
        visited(i) = 1;
        neighbors = find(D(:,i) <= eps);
        if length(neighbors) >= minPts 
            % We found a new cluster
            K = K + 1;
            [visited,cluster] = expand(X,i,neighbors,K,eps,minPts,D,visited,cluster);
        end
    end
end
model.Xtrain = X;
model.clusters = cluster;
end

function [visited,cluster] = expand(X,i,neighbors,K,eps,minPts,D,visited,cluster)
cluster(i) = K;
ind = 0;
while 1
    ind = ind+1;
    if ind > length(neighbors)
        break;
    end
    n = neighbors(ind);
    cluster(n) = K;
    
    if ~visited(n)
        visited(n) = 1;
        neighbors2 = find(D(:,n) <= eps);
        if length(neighbors2) >= minPts
            neighbors = [neighbors;setdiff(neighbors2,neighbors)];
        end
    end
    
    if size(X,2) == 2
        % Make plot
        clf;hold on;
        colors = getColorsRGB;  
        symbols = getSymbols;
        h = plot(X(cluster==0,1),X(cluster==0,2),'.');
        set(h,'Color',[0 0 0]);
        for k = 1:K
            h = plot(X(cluster==k,1),X(cluster==k,2),'.');
            set(h,'Color',colors(k,:),'Marker',symbols{k},'MarkerSize',12);
        end
        pause(.01);
    end
    
end
end