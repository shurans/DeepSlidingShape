function genPCindexList(id)
%cd /n/fs/modelnet/deepDetect/code/detector_3d/compute_feature
%/n/fs/vision/ionicNew/starter.sh genPCindexList 4500mb 165:00:00 1 300 1 

load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat')

PCindexList_dir = '/n/fs/modelnet/deepDetect/PCindexList/';
seg_dir ='/n/fs/modelnet/deepDetect/seg/';
allPath = [trainSeq,testSeq];
Space.s = 0.02;
addpath /n/fs/modelnet/SUN3DV2/prepareGT/
setup_benchmark;
if ~exist('id','var')
    imageNums = 1:length(allPath);
else
    imageNums =id:300:length(allPath);
end

for imageNum= imageNums
    
     fullpath2seq = allPath{imageNum};
     
     load([fullpath2seq 'data.mat']);
     tosavepath = fullfile(PCindexList_dir,[data.sequenceName '.bin']);
     
     if exist(tosavepath,'file')
        continue;
     end
     display(tosavepath);
     load(fullfile(seg_dir,data.sequenceName),'room')
     [rgb,points3d,imgZ]=read3dPoints(data);
     
     % remove outside points 
     points3d_align = [[room.Rot(1:2,1:2)*points3d(:,[1,2])']', points3d(:,3)];
     outside = points3d_align(:,1)<room.minX|points3d_align(:,1)>room.maxX|...
              points3d_align(:,2)<room.minY|points3d_align(:,2)>room.maxY|...
              points3d_align(:,3)<room.minZ|points3d_align(:,3)>prctile(points3d(:,3),90);

     points3d(outside,:) = NaN;
     % remove outside points 
     Range = nanmin(points3d);     
     xind  = floor((points3d(:,1)-Range(1))/Space.s);
     yind  = floor((points3d(:,2)-Range(2))/Space.s);
     zind  = floor((points3d(:,3)-Range(3))/Space.s);
     
     w = max(xind);
     d = max(yind);
     h = max(zind);
     
     [x,y,z] = ndgrid(0:w,0:d,0:h);
     startInd= 1;
     endInd = 1;
     star_end_indx_data = zeros(2,length(x(:)));
     %allPointsLinearInd = zeros(1,length(points3d));
     %colort = zeros(1,length(points3d));
     pc_lin_indx_data = [];
     for i =1:length(x(:))
         sel = xind==x(i)&yind==y(i)&zind==z(i);
         %colort(sel) =1;
         count = sum(sel);
         if count>0
            endInd = endInd+count;
            linearInd =x(i)*d*h+y(i)*h+z(i);
            star_end_indx_data(:,linearInd+1) = [startInd;endInd];
            pc_lin_indx_data(startInd:endInd-1) = find(sel);
            startInd = endInd;
         end
     end
     
     tosavepath = fullfile(PCindexList_dir,[data.sequenceName '.bin']);
     ind = find(tosavepath =='/');
     mkdir(tosavepath(1:ind(end)));
     
     points3d(isnan(points3d)) =0;
     allPoints = [points3d';rgb'];
     gridsize = [w,d,h];
     
     numidx = length(star_end_indx_data);
     numpc = length(pc_lin_indx_data);
     fid = fopen(tosavepath,'wb');
     fwrite(fid,int32(numidx),'int32');
     fwrite(fid,int32(numpc),'int32');
     fwrite(fid,int32(gridsize),'int32');
     fwrite(fid,int32([size(imgZ,1),size(imgZ,2)]),'int32');
     
     
     fwrite(fid,int32(star_end_indx_data),'int32');
     fwrite(fid,int32(pc_lin_indx_data),'int32');
     fwrite(fid,single([Range Space.s]),'single');
     fwrite(fid,single(allPoints),'single');
     fclose(fid);
end
if 0 
    fid = fopen('/Users/shuran/Dropbox/deepDetect/NYU0197.bin');
    %fid = fopen('debug.bin','rb');
    numidx = fread(fid,1,'int32');
    numpc = fread(fid,1,'int32');
    gridsize = fread(fid,3,'int32');
    imsize = fread(fid,2,'int32');
    
    star_end_indx_data = fread(fid,2*numidx,'int32');
    pc_lin_indx_data = fread(fid,numpc,'int32');
    range = fread(fid,3,'single');
    allPoints = fread(fid,6*imsize(1)*imsize(2),'single');
end
%save('PCindexList_debug')
if 0
    figure,
    for  i =1:length(x(:))
        linearInd =x(i)*d*h+y(i)*h+z(i);
        startInd = indMaxtrix(1,linearInd+1);
        endInd = indMaxtrix(2,linearInd+1);
        if endInd>0
            hold on;
            vis_point_cloud(allPoints(1:3,...
                pc_lin_indx_data(startInd:endInd-1))',...
                allPoints(4:6,pc_lin_indx_data(startInd:endInd-1))');
            pause;
        end
    end
end
end


