function draw_3d(XYZ,rgb,bb_3d,bb_3d_t,ptsSelect,s,color)
if ~isempty(XYZ)
    X = XYZ(10:s:end,1);
    Y= XYZ(10:s:end,2);
    Z = XYZ(10:s:end,3);
    RGB = rgb(10:s:end,:);
    scatter3(X(:),Y(:),Z(:),(6*(s-1)+1)*ones(size(X(:),1),1),double(RGB),'filled');  
end
hold on;
if ~isempty(bb_3d),
    for i=1:min(size(bb_3d,1))
        plotcube(bb_3d(i,4:6),bb_3d(i,1:3),0.3,color);
        hold on;
        if size(bb_3d,2)>6
           text(bb_3d(i,1)+bb_3d(i,4),bb_3d(i,2)+bb_3d(i,5),bb_3d(i,3)++bb_3d(i,6),num2str(round(bb_3d(i,7)*1000)/1000),'background','g');
        end
    end
end
if ~isempty(bb_3d_t)
   plotcube(bb_3d_t(1,4:6),bb_3d_t(1,1:3),0.1,[1 1 0]);
end
if ~isempty(ptsSelect)
   scatter3(ptsSelect(1,1:s:end)',ptsSelect(2,1:s:end)',ptsSelect(3,1:s:end)',ones(size(ptsSelect(1,1:s:end),2),1),ones(size(ptsSelect(1,1:s:end),2),1)*[0 1 0],'filled');
end
axis equal
axis tight

end 