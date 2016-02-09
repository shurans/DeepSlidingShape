function isIntightBBall = ptsInTightBB(points,BBTightAll)
        isIntightBBall =zeros(size(points,1),1);
        for i =1:length(BBTightAll)
            BBTight = BBTightAll(i);
            corners = get_corners_of_bb3d(BBTight);
            corners2d = corners(1:4,1:2);
            inTightBB2d = inpolygon(points(:,1),points(:,2),corners2d(:,1),corners2d(:,2));
            isIntightBB = inTightBB2d&points(:,3)>min(corners(1,3),corners(5,3))&points(:,3)<max(corners(1,3),corners(5,3));
            isIntightBBall = isIntightBBall|isIntightBB;
        end
        
end