load cities.mat

% categories: each row is a category
% names: each row is a city
% ratings: each row is a city, each col is a category

%% Run KNN

X = ratings;
K = 3;
limit = 10;

[N,D] = size(X);    % [rows,columns] of X

Distances = X.^2 * ones(D,N) + ones(N,D) * (X').^2 - 2 * X * (X');  % Each row is the distances of Xi from X1 to Xn

Dk = zeros(N,1);    % Average distance to K nearest neighbors
Nk = zeros(N,K);    % K nearest neighbors

outliernessScores = zeros(N,1);

% For each row of distances, get KNN
for i = 1:N
    % Sort a row of distances from small to large
    [sDistances, sIndex] = sort(Distances(:,i));
    
    % Compute the average distance from Xi to each K nearest neighbors
    Dk(i) = mean(sDistances(2:K+1,:));
    
    % Remember the K nearest neighbors
    Nk(i,:) = sIndex(2:K+1,:);
end

for i = 1:N
    % Calculate the outlierness score
    outliernessScores(i) = Dk(i) / mean(Dk(Nk(i,:)));
end

%% Sort Outlierness Scores
[sScores, sIndex] = sort(outliernessScores, 'descend');

scores = sScores(1:limit);
cityIndicies = sIndex(1:limit);
cities = names(cityIndicies,:);

for j = 1:size(cities,1)
    fprintf('%s\t%f\n', strtrim(cities(j,:)), scores(j));
end