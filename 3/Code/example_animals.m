%% Animals with attributes data
load animals.mat

%% K-Means clustering
K = 5;
model = clusterKmeans(X,K);

for k = 1:max(model.clusters)
    fprintf('Cluster %d: ',k);
    fprintf('%s ',animals{model.clusters==k});
    fprintf('\n');
end