load cities.mat

% categories: each row is a category
% names: each row is a city
% ratings: each row is a city, each col is a category

%% Run PCA
[n,d] = size(ratings);

X = standardizeCols(ratings);

[U,S,V] = svd(X);
k = 2;

% Slide
W = V(:,1:k)';
Z = X*W';

%% Set Dimensions
% The categories being compared
dimX = 1;
dimY = 2;

%% Plotting
figure(1);
scatter(Z(:,dimX),Z(:,dimY),'.');
labelX = strtrim(categories(dimX,:));
labelY = strtrim(categories(dimY,:));
xlabel(labelX);
ylabel(labelY);
gname(names);