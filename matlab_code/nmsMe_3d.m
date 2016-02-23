function [pick,suppressor,overlapall] = nmsMe_3d(boxes, overlap)
% Non-maximum suppression. (FAST VERSION)
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected
% detection.
% NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
% but an inner loop has been eliminated to significantly speed it
% up in the case of a large number of boxes

% modified based on Tomasz Malisiewicz's esvm code


if isempty(boxes)
  pick = [];
  suppressor =[];
  overlapall =[];
  return;
end
if size(boxes,1)<2
    suppressor =[];
    overlapall =[];
    pick =1;
    return;
end
x1 = boxes(:,1);
y1 = boxes(:,2);
z1 = boxes(:,3);
x2 = boxes(:,4);
y2 = boxes(:,5);
z2 = boxes(:,6);
s = boxes(:,7);

volume = (x2-x1) .* (y2-y1) .* (z2-z1);
[~, I] = sort(s);

pick = s*0;
suppressor = s*0;
overlapall =s*0;
counter = 1;
while ~isempty(I)
  
  last = length(I);
  i = I(last);  
  pick(counter) = i;
  
   
  xx1 = max(x1(i), x1(I(1:last-1)));
  yy1 = max(y1(i), y1(I(1:last-1)));
  zz1 = max(z1(i), z1(I(1:last-1)));
  xx2 = min(x2(i), x2(I(1:last-1)));
  yy2 = min(y2(i), y2(I(1:last-1)));
  zz2 = min(z2(i), z2(I(1:last-1)));
  
  w = max(0.0, xx2-xx1);
  h = max(0.0, yy2-yy1);
  d = max(0.0, zz2-zz1);
  inter = w.*h.*d;
  o = inter ./ (volume(i)+volume(I(1:last-1))-inter);
  
  idx = [last; find(o>overlap)];
  suppressor(I(idx)) = i;
  overlapall(I(idx)) = [1;o(find(o>overlap))];
  I(idx) = [];
  
  counter = counter + 1;
end

pick = pick(1:(counter-1));
end