function idx = drawLine(x1,y1,x2,y2,width,height,Linewidth)
        if ~exist('Linewidth','var')
            Linewidth =7;
        end

        idx =[drawLine_helper(x1,y1,x2,y2,width,height)];
        iter = floor(Linewidth/2);
        for i =1:iter
            idx =[idx drawLine_helper(x1+iter,y1+iter,x2+iter,y2+iter,width,height)];
            idx =[idx drawLine_helper(x1-iter,y1-iter,x2-iter,y2-iter,width,height)];
        end
end
function idx = drawLine_helper(x1,y1,x2,y2,width,height)
    x1 = round(x1(:));
    y1 = round(y1(:));
    x2 = round(x2(:));
    y2 = round(y2(:));
    nLines = length(x1);
    if length(y1) ~= nLines || length(x2) ~= nLines || length(y2) ~= nLines,
        error('inconsistent input size');
    end
    
    idx = [];
    for i = 1:nLines,
        if abs(x2(i)-x1(i)) > abs(y2(i)-y1(i)),
            lineX = min(x1(i),x2(i)):max(x1(i),x2(i));
            lineY = round(y1(i) + (lineX-x1(i)) * (y2(i)-y1(i)) / (x2(i)-x1(i)));
        elseif y1(i) ~= y2(i),
            lineY = min(y1(i),y2(i)):max(y1(i),y2(i));
            lineX = round(x1(i) + (lineY-y1(i)) * (x2(i)-x1(i)) / (y2(i)-y1(i)));
        else
            lineX = x1(i);
            lineY = y1(i);
        end
        isValid = lineX >= 1 & lineX <= width & lineY >= 1 & lineY <= height;
        idx = [idx sub2ind([height width],lineY(isValid),lineX(isValid))];
    end
end