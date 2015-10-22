%% Amazon Recommendation
load newsgroups.mat

%% Cosine Similarity

K = 5;
model = recommender(X,K);

for j = 1:5
    wordNumbers = model.predict(model,j);
    
    fprintf('Word: %s \n',wordlist{j});
    fprintf('Related words: ');
    fprintf('%s ',wordlist{wordNumbers});
    fprintf('\n');
end