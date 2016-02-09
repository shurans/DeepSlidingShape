function [minZ,maxZ,minX,maxX,minY,maxY,wall_floor] = getRoom(points3d,imageSeg,Rot,show)
points3d_align = [[Rot(1:2,1:2)*points3d(:,[1,2])']', points3d(:,3)];

mhaX = reshape(points3d_align(:,1),size(imageSeg));
mhaY = reshape(points3d_align(:,2),size(imageSeg));
mhaZ = reshape(points3d_align(:,3),size(imageSeg));

numSeg = max(imageSeg(:));
imageSeg(imageSeg(:)==0)= numSeg+1;
thresholdArea = 0.001;


for segID=1:numSeg
    roi = imageSeg==segID;
    
    if sum(roi(:)) > thresholdArea * numel(imageSeg)
        planeX = mhaX(roi(:));  
        planeY = mhaY(roi(:));  
        planeZ = mhaZ(roi(:));  
        stdXYZ =[std(planeX,1) std(planeY,1) std(planeZ,1)];
        [~,minIdx(segID)]=min(stdXYZ);
    else
        minIdx(segID) = 0;
    end
end
minIdx(numSeg+1)=0;


roi = minIdx(imageSeg)==1;
maxX = max(mhaX(roi(:)));
minX = min(-1,min(mhaX(roi(:))));


roi = minIdx(imageSeg)==2;
maxY = max(mhaY(roi(:)));
minY = min(-1,min(mhaY(roi(:))));



roi = minIdx(imageSeg)==3;
maxZ = max(mhaZ(roi(:)));
minZ = min(0,min(mhaZ(roi(:))));


tile = 1;
if isempty(maxZ)||maxZ<prctile(points3d_align(:,3),100-tile), maxZ = prctile(points3d_align(:,3),100-tile);end
if isempty(minZ)||minZ>prctile(points3d_align(:,3),tile), minZ = prctile(points3d_align(:,3),tile);end
if isempty(maxX)||maxX<prctile(points3d_align(:,1),100-tile), maxX = prctile(points3d_align(:,1),100-tile);end
if isempty(minX)||minX>prctile(points3d_align(:,1),tile), minX = prctile(points3d_align(:,1),tile);end
if isempty(maxY)||maxY<prctile(points3d_align(:,2),100-tile), maxY = prctile(points3d_align(:,2),100-tile);end
if isempty(minY)||minY>prctile(points3d_align(:,2),tile), minY = prctile(points3d_align(:,2),tile);end

wall_floor = zeros(size(imageSeg,1),size(imageSeg,2));
wall_floor(abs(mhaX-maxX)<0.1&minIdx(imageSeg)==1) = 1;
wall_floor(abs(mhaX-minX)<0.1&minIdx(imageSeg)==1) = 2;
wall_floor(abs(mhaY-maxY)<0.1&minIdx(imageSeg)==2) = 3;
wall_floor(abs(mhaZ-minZ)<0.1&minIdx(imageSeg)==3) = 4;
wall_floor(abs(mhaZ-maxZ)<0.1&minIdx(imageSeg)==3) = 5;

%% visualization 
if show
floor = [...
minX minY minZ
maxX minY minZ
maxX maxY minZ
minX maxY minZ]';

wallX = [...
maxX minY minZ
maxX maxY minZ
maxX maxY maxZ
maxX minY maxZ]';

wallY = [...
minX maxY minZ
maxX maxY minZ
maxX maxY maxZ
minX maxY maxZ]';


figure
clf
vis_point_cloud(points3d_align,[],10,10000); 
figure
clf
vis_point_cloud(points3d_align(wall_floor(:)==0,:),[],10,10000); 

xlabel('x'); ylabel('y'); zlabel('z');
hold on; patch(wallX(1,:),wallX(2,:),wallX(3,:),'r')
hold on; patch(wallY(1,:),wallY(2,:),wallY(3,:),'g')
hold on; patch(floor(1,:),floor(2,:),floor(3,:),'b')
alpha(0.6);
end
%}