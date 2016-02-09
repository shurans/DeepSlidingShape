function [pointCountIntegral,pointCount] =getIntegralPtCount(bincellDim,cellIdx)
pointCount = zeros(bincellDim);
[ind1,ind2,ind3]=ndgrid(1:bincellDim(1),1:bincellDim(2),1:bincellDim(3));
for i=1:length(ind1(:))
    ptsIncellidx = cellIdx(:,1)==ind1(i)&cellIdx(:,2)== ind2(i)&cellIdx(:,3)==ind3(i);
    pointCount(ind1(i),ind2(i),ind3(i),1:end)= sum(ptsIncellidx(:));
end
pointCountIntegral = cumsum(cumsum(cumsum(pointCount,1),2),3);
end