toload_dir = '/n/fs/modelnet/deepDetect/proposal/anchorBox';
load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat')
addpath /n/fs/modelnet/SUN3DV2/prepareGT/
setup_benchmark;
load('./external/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat');

allPath = [testSeq,trainSeq];
seqnames = cell(1,length(allPath));
for i =1:length(allPath)
    seqnames{i} = getSequenceName(allPath{i});
end

allRoomBox = [];
for imageNum= 1:length(allPath)
    %load(fullfile(toload_dir,seqnames{imageNum}),'Space');
    [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
    data = SUNRGBDMeta(ind);
    
    [rgb,points3d,imgZ]=read3dPoints(data);
    pt = 2;
    outsideP = points3d(:,1)<prctile(points3d(:,1),pt)|points3d(:,1)>prctile(points3d(:,1),100-pt)|...
                points3d(:,2)<prctile(points3d(:,2),pt)|points3d(:,2)>prctile(points3d(:,2),100-pt)|...
                points3d(:,3)<prctile(points3d(:,3),pt)|points3d(:,3)>prctile(points3d(:,3),90);

    points3d(outsideP,:) = [];
    Range  = nanmin(points3d); 
    Range2 = nanmax(points3d);
     
    corner = [Range,Range2-Range];
    allRoomBox = [allRoomBox;corner];
    
    
    %{
    [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
    data = SUNRGBDMeta(ind);
    [rgb,points3d,imgZ]=read3dPoints(data);
    figure(2)
    clf
    draw_3d(points3d,rgb,corner,[],[],10,[0,1,0]);
    view(0,90)
    pause;
    %}
end


figure
hold on;
for i =1:size(allRoomBox,1)
    draw_3d([],[],allRoomBox(i,:),[],[],1,rand(1,3));
end
