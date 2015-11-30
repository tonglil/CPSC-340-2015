function [] = classifier2Dplot(X,y,k,model)
increment = 500;

colors = getColorsRGB;
symbols = getSymbols;

figure(1);clf;hold on
for c = 1:k
    h = plot(X(y==c,2),X(y==c,3),'b.');
    set(h,'Color',colors(c,:),'Marker',symbols{c});
end

domain1 = xlim;
domain1 = domain1(1):(domain1(2)-domain1(1))/increment:domain1(2);
domain2 = ylim;
domain2 = domain2(1):(domain2(2)-domain2(1))/increment:domain2(2);

d1 = repmat(domain1',[1 length(domain1)]);
d2 = repmat(domain2,[length(domain2) 1]);
t = length(d1(:));

% Get predictions
vals = model.predict(model,[ones(t,1) d1(:) d2(:)]);
if size(vals,1) ~= length(d1(:))
    error('Output of model.predict should have T rows');
elseif size(vals,2) ~= 1
    error('Output of model.predict should have 1 column');
end
z = reshape(vals,size(d1));

% For plotting purposes, remove classes that don't occur
u = unique(z(:));
for c = k:-1:1
    if ~any(z(:)==c)
        z(z > c) = z(z > c)-1;
    end
end
contourf(d1,d2,z,1:max(z(:)),'k');
colormap(colors(u,:)/2);

for c = 1:k
    h = plot(X(y==c,2),X(y==c,3),'b.');
    set(h,'Color',colors(c,:),'Marker',symbols{c});
end

