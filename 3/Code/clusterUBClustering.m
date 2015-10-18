function [model] = clusterUBClustering(X,K,nModels)

[N,D] = size(X);

for m = 1:nModels
    model.subModel{m} = clusterKmeans(X,K);
end

for m = 1:nModels
    clusters(:,m) = model.subModel{m}.predict(model.subModel{m},X);
end
clusters = mode(clusters,2);
model.clusters = clusters;

% If we only have two features, make a colored scatterplot
if D == 2
    clf;hold on;
    colors = getColorsRGB;
    for k = 1:K
        plot(X(clusters==k,1),X(clusters==k,2),'o','Color',.75*colors(k,:),'MarkerSize',5,'MarkerFaceColor',.75*colors(k,:));
    end
end
end


