function [Z] = visualizeISOMAP(X,k,names)
[n,d] = size(X);

% Compute all distances
D = X.^2*ones(d,n) + ones(n,d)*(X').^2 - 2*X*X';
D = sqrt(abs(D));

knn = 2;
G = zeros(n);    % Adjacency matrix of distances to K nearest neighbors

% For each column of distances, get KNN
for i = 1:n
    % Sort a column of distances from small to large
    [sDistances, sIndex] = sort(D(:,i));
    
    % Get the distance to the K nearest neighbors
    distances = sDistances(2:knn+1);
    neighbors = sIndex(2:knn+1);

    % Each column (~= 0) in the matrix represents the weight of edges to nodes
    G(neighbors,i) = distances;
    % Make the matrix symmetrical
    G(i,neighbors) = distances;
end

% Reset all distances
D = zeros(n);

maxD = 0;

% For each node, get the shortest distance to another node
for i = 1:n
    for j = 1:n
        % The distance to itself is 0
        if j ~= i
            [cost, ~] = dijkstra(G,i,j);
            
            if (~isinf(cost) && cost > maxD)
                maxD = cost;
            end
            
            D(i,j) = cost;
        end
    end
end

% If the distance is infinite, set it to the max
D(isinf(D)) = maxD;

% Run MDS

% Initialize low-dimensional representation with PCA
[U,S,V] = svd(X);
W = V(:,1:k)';
Z = X*W';

Z(:) = findMin(@stress,Z(:),500,0,D,names);

end

function [f,g] = stress(Z,D,names)
n = length(D);
k = numel(Z)/n;

Z = reshape(Z,[n k]);

f = 0;
g = zeros(n,k);
for i = 1:n
    for j = i+1:n
        % Objective Function
        Dz = norm(Z(i,:)-Z(j,:));
        s = D(i,j) - Dz;
        f = f + (1/2)*s^2 / D(i,j);
        
        % Gradient
        df = s / D(i,j);
        dgi = (Z(i,:)-Z(j,:))/Dz;
        dgj = (Z(j,:)-Z(i,:))/Dz;
        g(i,:) = g(i,:) - df*dgi;
        g(j,:) = g(j,:) - df*dgj;
    end
end
g = g(:);

% Make plot if using 2D representation
if k == 2
    figure(3);
    clf;
    plot(Z(:,1),Z(:,2),'.');
    hold on;
    for i = 1:n
        text(Z(i,1),Z(i,2),names(i,:));
    end
    pause(.01)
end
end