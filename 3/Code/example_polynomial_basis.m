% Load data
load basisData.mat

split = 250;

% Split the data into training and test sets
xTraining = X(1:split,:);
yTraining = y(1:split,:);
xTest = X(split+1:end,:);
yTest = y(split+1:end,:);

degree = 20;

ASETraining = zeros(degree+1,1);
ASETest = zeros(degree+1,1);

for j = 0:degree
    % Fit least-squares with polynomial basis estimator
    model = leastSquaresBasis(xTraining,yTraining,j);

    % Training error
    yHat = model.predict(model,xTraining);
    ASETraining(j+1) = sum((yHat - yTraining).^2) / 250;
    
    % Validation error
    yHat = model.predict(model,xTest);
    ASETest(j+1) = sum((yHat - yTest).^2) / 250;

    fprintf('Degree : %i \n',j);
    fprintf('Training error: %f \n',ASETraining(j+1));
    fprintf('Validation error: %f \n',ASETest(j+1));
end

plot(0:degree,ASETraining,'-go');
plot(0:degree,ASETest,'-bx');
title('Training Error vs Test Error');
legend('Test Error','Training Error');
hold on;

% plot(0:degree, abs(ASETraining - ASETest)./ASETraining, '-r*');
% legend('(trainMSE - validationMSE) / trainMSE');