% Load data
load basisData.mat

% Create the figure
figure(1);

degree = 8;

for j = 0:degree
    % Draw and label each subplot
    subplot(3,3,j+1);
    plot(X,y,'b.');
    string = sprintf('degree %i', j);
    title(string);
    hold on;
    
    % Fit least-squares with polynomial basis estimator
    model = leastSquaresBasis(X,y,j);

    % Draw model prediction
    Xsample = [min(X):.1:max(X)]';
    yHat = model.predict(model,Xsample);
    plot(Xsample,yHat,'g-'); 
end