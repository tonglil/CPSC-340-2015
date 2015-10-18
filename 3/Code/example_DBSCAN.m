%% Clustering
load clusterData.mat

%% Density-Based Clustering
eps = 1;
minPts = 3;
model = clusterDBcluster(X,eps,minPts);
title('Densty-Based clustering');