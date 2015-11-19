function [model] = recommendUserItemMean(X,y)

n = max(X(:,1));
d = max(X(:,2));
nRatings = size(X,1);

bu = zeros(n,1);
Z = zeros(n,1);
for i = 1:nRatings
    bu(X(i,1)) = bu(X(i,1)) + y(i); % Add the rating to the running sum
    Z(X(i,1)) = Z(X(i,1)) + 1; % Add one to the number of ratings for user
end
bu = bu./Z; % Normalize counts by number of ratings

bm = zeros(d,1);
Z = zeros(d,1);
for i = 1:nRatings
    bm(X(i,2)) = bm(X(i,2)) + y(i); % Add the rating to the running sum
    Z(X(i,2)) = Z(X(i,2)) + 1; % Add one to the number of ratings for user
end
bm = bm./Z; % Normalize counts by number of ratings

model.bu = bu;
model.bm = bm;
model.predict = @predict;
end

function [y] = predict(model,X)
t = size(X,1);
bu = model.bu;
bm = model.bm;

y = zeros(t,1);
for i = 1:t
    u = X(i,1);
    m = X(i,2);
    y(i) = bu(u)/2 + bm(m)/2; % Take the average between user and movie ratings
end
end