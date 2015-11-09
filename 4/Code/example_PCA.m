load animals.mat

[n,d] = size(X);
X = standardizeCols(X);

figure(1);
imagesc(X);
figure(2);
plot(X(:,1),X(:,2),'.');
gname(animals);
