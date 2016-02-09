function draw_2d(rgb,BB_2d,isTp,gtBB)
 if max(rgb(:))>1.1, 
     rgb = double(rgb)/255;
 end
     imshow(rgb);
 if isempty(BB_2d),return;end
 hold on;
 if isempty(isTp)
     isTp =ones(size(BB_2d,1),1);
 end
 for i=1:size(BB_2d,1)
     try rectangle('Position', BB_2d(i,1:4),'edgecolor',[~isTp(i),isTp(i),0]);end
     if size(BB_2d,2)>4
          try text(BB_2d(i,1),BB_2d(i,2),num2str(BB_2d(i,5)),'BackgroundColor',[~isTp(i),isTp(i),0]);end
     end
 end
 if exist('gtBB','var')&&~isempty(gtBB)
     for i=1:length(gtBB)
        rectangle('Position', gtBB(i).gtBb2D(1,1:4),'edgecolor','y');
        %text(gtBB(i).gtBb2D(1,1),gtBB(i).gtBb2D(1,2),num2str(i),'edgecolor','y');
     end
 end
 hold off;
end