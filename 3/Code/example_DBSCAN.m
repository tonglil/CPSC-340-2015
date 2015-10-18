%% Clustering
load clusterData.mat

%% Density-Based Clustering
eps = 14;
minPts = 3;
model = clusterDBcluster(X,eps,minPts);
title('Densty-Based clustering');