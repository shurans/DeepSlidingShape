uniquename = unique({groundtruth.classname});
for i =1:length(uniquename)
    count(i) = sum(ismember({groundtruth.classname},uniquename{i}));
end
[num,ind] = sort(count,'descend');
uniquename = uniquename(ind);