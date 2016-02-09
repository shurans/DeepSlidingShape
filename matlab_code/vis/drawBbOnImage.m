function img = drawBbOnImage(img,bb2dDraw,bbColor,Linewidth)
    if ~exist('Linewidth','var')
        Linewidth =3;
    end
    if size(bb2dDraw,2) < 17, error('wrong bb format.'); end
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
    
    p1 = [bb2dDraw(2*link(:,1)-1)' bb2dDraw(2*link(:,1))'];
    p2 = [bb2dDraw(2*link(:,2)-1)' bb2dDraw(2*link(:,2))'];
    width = size(img,2);
    height = size(img,1);
    idx = drawLine(p1(:,1),p1(:,2),p2(:,1),p2(:,2),size(img,2),size(img,1),Linewidth);
    img(sub2ind([height*width 3],idx,1*ones(size(idx)))) = bbColor(1);
    img(sub2ind([height*width 3],idx,2*ones(size(idx)))) = bbColor(2);
    img(sub2ind([height*width 3],idx,3*ones(size(idx)))) = bbColor(3);

end