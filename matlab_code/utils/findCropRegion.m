function crop = findCropRegion(mask)

[ylist, xlist] = ind2sub(size(mask), find(mask));
xmin = min(xlist); xmax = max(xlist);
ymin = min(ylist); ymax = max(ylist);
crop = [xmin,ymin, xmax-xmin, ymax-ymin];

end