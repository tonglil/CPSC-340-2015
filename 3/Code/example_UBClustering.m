%% Clustering
load clusterData.mat

%% UBClustering
K = 4;
nModels = 20;
model = clusterUBClustering(X,K,nModels);
cluster = model.clusters;

%% Make plot
clf;
hold on;

colors = getColorsRGB;
symbols = getSymbols;

h = plot(X(cluster==0,1),X(cluster==0,2),'.');
set(h,'Color',[0 0 0]);

for k = 1:K
    h = plot(X(cluster==k,1),X(cluster==k,2),'.');
    set(h,'Color',colors(k,:),'Marker',symbols{k},'MarkerSize',12);
end
title('UBClustering');