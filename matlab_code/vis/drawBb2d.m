function image = drawBb2d(image,bb2dX,bb2dY,color)
    link = [1,2;
            1,3;
            2,4;
            3,4;
            1,5;
            2,6;
            3,7;
            4,8;
            5,6;
            5,7;
            6,8;
            7,8];
    
    nBbs = size(bb2dX,1);
    width = size(image,2);
    height = size(image,1);
    for i = 1:nBbs,
        idx = drawLine(bb2dX(i,link(:,1)),bb2dY(i,link(:,1)),bb2dX(i,link(:,2)),bb2dY(i,link(:,2)),width,height);
        image(sub2ind([height*width 3],idx,1*ones(size(idx)))) = color(i,1);
        image(sub2ind([height*width 3],idx,2*ones(size(idx)))) = color(i,2);
        image(sub2ind([height*width 3],idx,3*ones(size(idx)))) = color(i,3);
    end
end