function [model] = recommendItemMean(X,y)

n = max(X(:,1));
d = max(X(:,2));
nRatings = size(X,1);

b = zeros(d,1);
Z = zeros(d,1);
for i = 1:nRatings
    b(X(i,2)) = b(X(i,2)) + y(i); % Add the rating to the running sum
    Z(X(i,2)) = Z(X(i,2)) + 1; % Add one to the number of ratings for user
end
b = b./Z; % Normalize counts by number of ratings

model.b = b;
model.predict = @predict;
end

function [y] = predict(model,X)
t = size(X,1);
b = model.b;

y = zeros(t,1);
for i = 1:t
    y(i) = b(X(i,2));
end
end
