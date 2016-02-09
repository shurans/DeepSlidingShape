function [imageall,imgh,imgw] = vis_applyBbMaskTightBB(rgb,points,data,tightBB,gtBB,maskColor)
    maskRatio = 1/3;
    
    % remove nan points 
    isInvalid = sum(isnan(points),2) > 0;
    points(isInvalid,:) = [];
    rgb(isInvalid,:) = [];
    % find points inside bb
    isInBb = ptsInTightBB(points,tightBB);
             
    % mask points inside with mask color
    rgb(isInBb,:) = bsxfun(@plus, (1-maskRatio)*rgb(isInBb,:), maskRatio*maskColor);
    
    % project point to 2D space
    points2d = round(project3dPtsTo2d(points,data.Rtilt,[1,1],data.K));
    %points2d(:,1) = points2d(:,1) - min(points2d(:,1)) + 1;
    %points2d(:,2) = points2d(:,2) - min(points2d(:,2)) + 1;
    image = 255*ones(max(points2d(:,2)),max(points2d(:,1)),3);
    image(sub2ind(size(image),points2d(:,2),points2d(:,1),1*ones(size(points2d,1),1))) = rgb(:,1);
    image(sub2ind(size(image),points2d(:,2),points2d(:,1),2*ones(size(points2d,1),1))) = rgb(:,2);
    image(sub2ind(size(image),points2d(:,2),points2d(:,1),3*ones(size(points2d,1),1))) = rgb(:,3);
    
    [imgh,imgw,k] =size(image); 
    
    %top view of whole
   [topview_oth,minbound2d,minbound3d,imageScale,camDist]  =renderPointCloud_orthogonal(points,rgb,[],imgh,imgh,[]);
    % draw gt box on topview_oth
    for i=1:length(gtBB)      
        topview_oth =drawtopviewBBstructBB(gtBB(i),minbound2d,imageScale,topview_oth,[1,1,0],image);
    end  
    % draw tight bb on topview
    for i=1:length(tightBB)    
        topview_oth =drawtopviewBBstructBB(tightBB(i),minbound2d,imageScale,topview_oth,maskColor,image);
    end
    gaps = 255*ones(size(image,1),5,3);
    imageall= [image,gaps,topview_oth];
    
%    imshow(imageall); pause;
%    [~,bb2dDraw] = project3dBbsTo2d(bb, R);
%    visualizeProjected3dBb(bb2dDraw);
end  

function out = roundwithboud(points2d,image_scale,size_d)
         out = min(max(round(points2d*image_scale),1),size_d);
end

function topview_oth =drawtopviewBBstructBB(BB,minbound2d,imageScale,topview_oth,BBcolor,image)
         gtbbpoints = get_corners_of_bb3d(BB);
        gtbbpoints([1,2,3,4,5,6,7,8],:) = gtbbpoints([8 4 5 1 7 3 6 2],:);
%        gtbbpoints(:,2) = gtbbpoints(:,2) - minbound3d(2) + camDist;
       points2d = [gtbbpoints(:,1),-1*gtbbpoints(:,2)];
       points2d(:,1) = points2d(:,1) - minbound2d(1) + 1;
       points2d(:,2) = points2d(:,2) - minbound2d(2) + 1;

        points2d(:,1) = min(max(round(points2d(:,1) * imageScale(1)),1),size(image,1));
        points2d(:,2) = min(max(round(points2d(:,2) * imageScale(2)),1),size(image,1));
%        points2d2Draw =points2d';
%        points2d2Draw =reshape(points2d2Draw(:),16,[]);
        bb2dX = transpose(reshape(points2d(:,1),8,[]));
        bb2dY = transpose(reshape(points2d(:,2),8,[]));
        topview_oth = drawBb2d(topview_oth,bb2dX,bb2dY,BBcolor);
end