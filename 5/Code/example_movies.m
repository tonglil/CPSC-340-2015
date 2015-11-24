%% Load data
load movies.mat

%% Global-average baseline

model = recommendGlobalMean(X,y);
yhat = model.predict(model,Xvalidate);
err = mean(abs(yhat - yvalidate));

fprintf('Average absolute error by using global average rating: %f\n',err);

%% User-average baseline

model = recommendUserMean(X,y);
yhat = model.predict(model,Xvalidate);
err = mean(abs(yhat - yvalidate));

fprintf('Average absolute error by using user average rating: %f\n',err);

%% Movie-average baselinel

model = recommendItemMean(X,y);
yhat = model.predict(model,Xvalidate);
err = mean(abs(yhat - yvalidate));

fprintf('Average absolute error by using movie average rating: %f\n',err);

%% User-average baseline

model = recommendUserItemMean(X,y);
yhat = model.predict(model,Xvalidate);
err = mean(abs(yhat - yvalidate));

fprintf('Average absolute error by using user+movie average rating: %f\n',err);

%% SVD Recommender

k = 10;
model = recommendSVD(X,y,k);
yhat = model.predict(model,Xvalidate);
err = mean(abs(yhat - yvalidate));

fprintf('Average absolute error by using user+movie average rating: %f\n',err);

fprintf('Done');