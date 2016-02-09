function [pick,suppressor,overlapall] = nms3d(bbstruct, overlap)

if isempty(bbstruct)
  pick = [];
  suppressor =[];
  overlapall =[];
  return;
end
if size(bbstruct,1)<2
    pick =1;
    suppressor =[];
    overlapall =[];
    return;
end

s = [bbstruct.conf];

[~, I] = sort(s);

pick = s*0;
suppressor = s*0;
overlapall =s*0;
counter = 1;
while ~isempty(I)
  
  last = length(I);
  i = I(last);  
  pick(counter) = i;
  
   
  o = bb3dOverlapCloseForm(bbstruct(I(1:last-1)),bbstruct(i));
  
  idx = [last; find(o>overlap)];
  suppressor(I(idx)) = i;
  overlapall(I(idx)) = [1;o(find(o>overlap))];
  I(idx) = [];
  
  counter = counter + 1;
end

pick = pick(1:(counter-1));
end