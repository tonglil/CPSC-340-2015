load cities.mat

% categories: each row is a category
% names: each row is a city
% ratings: each row is a city, each col is a category

[n,d] = size(ratings);
scores = zeros(n,d);

for i = 1:d
    cat = ratings(:,i);
    
    score = (cat - mean(cat)) / std(cat);
    
    outliers = find(score > 4);
    
    % Show the outliers that exist in a certain category
    if (size(outliers,1))
        label = categories(i,:);
        cities = names(outliers,:);
        
        % Print results
        fprintf('Category: %s\n', strtrim(label));
        for j = 1:size(cities,1)
            fprintf('- %s\n', strtrim(cities(j,:)));
        end
        fprintf('\n');
    end    
end