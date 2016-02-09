function dss_preparedata(id)

opt = dss_initPath();
seg_dir         = opt.seg_dir;
PCindexList_dir = opt.data_root;
load(fullfile(cnn3d_model.conf.SUNrgbd_toolbox,'Metadata/SUNRGBDMeta.mat'));

NYUonly =1;
if NYUonly
    load('./external/SUNRGBDtoolbox/traintestSUNRGBD/test_kv1NYU.mat');
    load('./external/SUNRGBDtoolbox/traintestSUNRGBD/train_kv1NYU.mat');
    allPath = [testSeq,trainSeq];
else
   load('./external/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
   allPath = [alltest,alltrain];
end

Space.s = 0.02;

if ~exist('id','var')
    imageNums = 1:length(allPath);
else
    imageNums =id:300:length(allPath);
end

seqnames = cell(1,length(allPath));
for i =1:length(allPath)
    seqnames{i} = getSequenceName(allPath{i});
end

for imageNum= imageNums
    [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
    data = SUNRGBDMeta(ind);
     
     tosavepath = fullfile(PCindexList_dir,[data.sequenceName '.bin']);

     if exist(tosavepath,'file')
        fprintf('skipng image %s...\n',tosavepath)
        continue;
     end
     fprintf('computing image %s...\n',tosavepath);
     load(fullfile(seg_dir,data.sequenceName),'room')
     [rgb,points3d,imgZ]=read3dPoints(data);

     % remove outside points 
     points3d_align = [[room.Rot(1:2,1:2)*points3d(:,[1,2])']', points3d(:,3)];
     outside = points3d_align(:,1)<room.minX|points3d_align(:,1)>room.maxX|...
              points3d_align(:,2)<room.minY|points3d_align(:,2)>room.maxY|...
              points3d_align(:,3)<room.minZ|points3d_align(:,3)>prctile(points3d(:,3),90);
    
     pt = 0.02;
     outsideP = points3d(:,1)<prctile(points3d(:,1),pt)|points3d(:,1)>prctile(points3d(:,1),100-pt)|...
                points3d(:,2)<prctile(points3d(:,2),pt)|points3d(:,2)>prctile(points3d(:,2),100-pt)|...
                points3d(:,3)<prctile(points3d(:,3),pt)|points3d(:,3)>prctile(points3d(:,3),90);

     points3d(outside|outsideP,:) = NaN;

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
     star_end_indx_data = [];
     star_end_lin_idx =[];
     pc_lin_indx_data = [];

     for i =1:length(x(:))
         sel = xind==x(i)&yind==y(i)&zind==z(i);
         count = sum(sel);
         if count>0
            endInd = endInd+count;
            linearInd =x(i)*d*h+y(i)*h+z(i);
            star_end_indx_data(:,end+1) = [startInd-1;endInd-1];
            star_end_lin_idx(:,end+1) = linearInd;
            pc_lin_indx_data(startInd:endInd-1) = find(sel);
            startInd = endInd;
         end
     end
     
     ind = find(tosavepath =='/');
     mkdir(tosavepath(1:ind(end)));
     grid_range = [w,d,h];


     fid = fopen(tosavepath,'wb');
     fwrite(fid,uint32(grid_range),'uint32');
     fwrite(fid,single(Range),'single');
     fwrite(fid,single(Space.s),'single');

     image = imread(data.rgbpath);
     RGB =  reshape(image, size(image,1)*size(image,2), 3)';
     Depth = imread(data.depthpath);
     Depth = bitor(bitshift(Depth,-3), bitshift(Depth,16-3));
     Depth = reshape(typecast(Depth(:)','uint8'),2,[]);
     fwrite(fid,[RGB;Depth],'uint8');

     fwrite(fid,uint32(numel(star_end_indx_data)),'uint32');
     fwrite(fid,uint32(star_end_indx_data),'uint32');

     fwrite(fid,uint32(numel(star_end_lin_idx)),'uint32');
     fwrite(fid,uint32(star_end_lin_idx),'uint32');

     fwrite(fid,uint32(numel(pc_lin_indx_data)),'uint32');
     fwrite(fid,uint32(pc_lin_indx_data),'uint32');
end
