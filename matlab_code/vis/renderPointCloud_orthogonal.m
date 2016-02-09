function [image,minbound2d,minbound3d,imageScale,camDist] = renderPointCloud_orthogonal(points,rgb,bb,imageHeight,imageWidth,bbsearch)
    % shift point cloud to origin, remember the minimum distance to camera (xz plane)
    image = ones(imageHeight,imageWidth,3);
    nPoints = size(points,1);
    camDist = min(points(:,2));
    center = median(points,1);
    if nPoints ==0,
        return;
    end
    
    if ~isempty(bbsearch)
    bb = [bb;bbsearch zeros(1,size(bb,2)-size(bbsearch,2))];
    end
    if ~isempty(bb),
        [bbVertices3d,bbConfs] = bb3dToVertices(bb);
        points = [points; bbVertices3d];
    end
    %points = bsxfun(@minus,points,center);
    
    % shift point cloud to its original center, and maintain the minimum distance to camera (xz plane)
   % points = bsxfun(@plus,points,center);
    minbound3d =min(points(1:nPoints,:));
    points(:,2) = points(:,2) - min(points(1:nPoints,2)) + camDist;
   
    % project point to 2D space
    points2d = [points(:,1),-1*points(:,2)];
    minbound2d =[min(points2d(1:nPoints,1)),min(points2d(1:nPoints,2))];
    
    points2d(:,1) = points2d(:,1) - min(points2d(1:nPoints,1)) + 1;
    points2d(:,2) = points2d(:,2) - min(points2d(1:nPoints,2)) + 1;
    
    if ~isempty(bb),
        bbVertices2d = points2d(nPoints+1:end,:);
        points2d = points2d(1:nPoints,:);
    end
    
    % generate projection image
    imageScale = [imageWidth/max(points2d(:,1)),imageHeight/max(points2d(:,2))];
    imageScale = repmat(min(imageScale),[1 2]);
    
    points2d(:,1) = min(max(round(points2d(:,1) * imageScale(1)),1),imageWidth);
    points2d(:,2) = min(max(round(points2d(:,2) * imageScale(2)),1),imageHeight);
    if ~isempty(bb),
        bbVertices2d(:,1) = bbVertices2d(:,1) * imageScale(1);
        bbVertices2d(:,2) = bbVertices2d(:,2) * imageScale(2);
        bb2dX = transpose(reshape(bbVertices2d(:,1),8,[]));
        bb2dY = transpose(reshape(bbVertices2d(:,2),8,[]));
    end
    
    
    image(sub2ind(size(image),points2d(:,2),points2d(:,1),1*ones(size(points2d,1),1))) = rgb(:,1);
    image(sub2ind(size(image),points2d(:,2),points2d(:,1),2*ones(size(points2d,1),1))) = rgb(:,2);
    image(sub2ind(size(image),points2d(:,2),points2d(:,1),3*ones(size(points2d,1),1))) = rgb(:,3);
    
    % draw bb
    if ~isempty(bb),
        image = drawBb2d(image,bb2dX,bb2dY,[0 1 1;0 0 1]);
    end
end