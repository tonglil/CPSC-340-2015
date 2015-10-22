function [model] = recommender(X,K)
model.X = X;
model.K = K;
model.predict = @predict;
end

function [wordNumbers] = predict(model,j)
X = model.X;
K = model.K;

[~,D] = size(X);
% D = words: each column is a word

% Initialize cosine similarities
cosSims = zeros(D,1);

% Xi
xi = X(:,j);
xiNorm = norm(xi);

% Find cosine similarity in relation to every other word
for i = 1:D;
    % Skip finding the cosine similarity for the word itself
    if i ~= j;
        % Xj
        xj = X(:,i);
        xjNorm = norm(xj);
        
        % Calculate the cosine similarity
        cosSims(i) = dot(xi',xj) / (xiNorm * xjNorm);
    end
end

[~, I] = sort(cosSims, 'descend');

% Return a vector of the first K words
wordNumbers = I(1:K);
end
