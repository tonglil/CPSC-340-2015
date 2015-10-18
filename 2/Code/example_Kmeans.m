% Clustering
load clusterData.mat

% K-Means Clustering
K = 4;
cleamodel = clusterKmeans(X,K);
title('K-Means clustering');