function image = renderPointCloud(points,rgb,bb,Rxyz,Rtilt,K,imageHeight,imageWidth)
    % shift point cloud to origin, remember the minimum distance to camera (xz plane)
    isInvalid = sum(isnan(points),2) > 0;
    points(isInvalid,:) = [];
    rgb(isInvalid,:) = [];
    
    image = ones(imageHeight,imageWidth,3);
    nPoints = size(points,1);
    camDist = min(points(:,2));
    center = median(points,1);
    
    if nPoints ==0,
        return;
    end
    
    if ~isempty(bb),
        [bbVertices3d,bbConfs] = bb3dToVertices(bb);
        points = [points; bbVertices3d];
    end
    points = bsxfun(@minus,points,center);
    
    % apply rotation matrix to point cloud
    points = Rxyz * [points ones(size(points,1),1)]';
    points = bsxfun(@rdivide,points(1:3,:)',points(4,:)');
    
    % shift point cloud to its original center, and maintain the minimum distance to camera (xz plane)
    points = bsxfun(@plus,points,center);
    points(:,2) = points(:,2) - min(points(1:nPoints,2)) + camDist;
    
    % project point to 2D space
    points2d = (project3dPtsTo2d(points,Rtilt,K));
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
        image = drawBb2d(image,bb2dX,bb2dY,[0 1 0]);
    end
end