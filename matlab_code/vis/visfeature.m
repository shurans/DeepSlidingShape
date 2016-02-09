fea_encode = 0;

fout = readTensor('/n/fs/modelnet/deepDetect/code/marvin/DSS/debug/data0962',1);
features = fout.value;
outputSize = size(features);
fdim =outputSize(4);
numofbox = 1;
%}
%{
fout = fopen('/Volumes/modelnet/deepDetect/code/marvin_test/DSS/debug/XYZimage_GPU.bin','rb');
XYZimage =fread(fout,480*640*3,'float');
XYZimage = reshape(XYZimage,[3,480,640]);
points2ply(sprintf('shape.ply',imageID), reshape(XYZimage,3,[]));
 %}


%{
fout = fopen('/Volumes/modelnet/deepDetect/code/marvin_test/DSS/debug/feature_renderSS.bin','rb');
outputSize = [30,30,30];
numofbox = 1;
%}
%fdim =3;
%features=fread(fout,[prod(outputSize)*fdim*numofbox],'float');
%fclose(fout);
%features = reshape(features,[outputSize(3),outputSize(2),outputSize(1),fdim,numofbox]);
%%
for imageID =1 % 1:numofbox
xyz = zeros(3,0);
fea = zeros(fdim,0);
for x=1:outputSize(1)
    for y=1:outputSize(2)
        for z=1:outputSize(3)
            c = reshape(features(z,y,x,:,imageID),fdim,[]);
            xyz(:,end+1) = [-x;y;z];
            %rgb(:,end+1) = c([3 2 1]);
            fea(:,end+1) = c;
        end
    end
end
fea = fea/100;
% points2ply(sprintf('shape_%d.ply',imageID), fea);
%%
% sel = abs(fea(4,:))>0.0;
% xyz = xyz(:,sel);
% fea = fea(:,sel);
% points2ply(sprintf('shape_%d.ply',imageID), xyz,fea([3,2,1],:));

%% ratio second ratio fea
dimselect =2;
fead = fea(dimselect,:);
%{
figure,%
p = patch(isosurface(reshape(fead,outputSize),0));
n = isonormals(reshape(fead,outputSize),p);
p.FaceColor = 'red';
p.EdgeColor = 'none';
camlight 
lighting gouraud;
%}

fprintf('max=%f\nmin=%f\n',max(fead(:)),min(fead(:)));
%hist(fead(:),50)
sel = (abs(fead)< max(abs(fead(:)))-0.0001)&abs(fead)>0;
xyz_sel = xyz(:,sel);
fead_sel = fead(:,sel);
fead_sel(xyz_sel(2,:)<40) = fead_sel(xyz_sel(2,:)<40)+ 0.02;
%fead_sel(fead_sel>-0.02&fead_sel<0) =fead_sel(fead_sel>-0.02&fead_sel<0) + 0.02;

points2ply(sprintf('%d_%d_%d.ply',fea_encode,imageID,dimselect), xyz_sel,reshape(getImagesc(fead_sel'),length(fead_sel),3)');
if 0
   fout = fopen('/Users/shuran/Dropbox/marvin/DSS/XYZimage.bin','rb');
   XYZ =fread(fout,[427*561*3],'float');
   XYZ = reshape(XYZ,3,427*561);
   depth = XYZ(3,:);
    points2ply(sprintf('shape_%d.ply',imageID), XYZ);
end
end
%%
labelsize = [19,53,53,26];
labelCPU_0 = readTensor_v2('/Volumes/modelnet/deepDetect/code/marvin/DSS/debug/labelCPU.tensor');
labelCPU_all{1} = labelCPU_0.value(:,:,:,:,:,1);

labelCPU_1 = readTensor_v2('/Volumes/modelnet/deepDetect/code/marvin/DSS/debug/labelCPU_1.tensor');
labelCPU_all{2} = labelCPU_1.value;
labelCPU = zeros(labelsize(end:-1:1));
for gid = 0:1
    idx = find(size_group==gid);
    for i =1:size(labelCPU_all{gid+1},4)
        labelCPU(:,:,:,idx(i)) = labelCPU_all{gid+1}(:,:,:,i);
    end
end

for i =1:19
    box = labelCPU(:,:,:,i);
    [indexes1, indexes2, indexes3] = ind2sub(size(box),find(box(:)>0));
    hold on;
    pts = bsxfun(@plus,[indexes3  indexes2  indexes1]*0.1,[-2.6,0.4,-1.5]);
    plot3(pts(:,1),pts(:,2),pts(:,3),'xr');
end

load('/Volumes/modelnet/deepDetect/RPNdata_mulit/SUNRGBD/kv1/NYUdata/NYU0003.mat')
[oscf,matchgt] = max(oscfM,[],2);
posbox = center_Idx_ALL(oscf>0.35,:);
negbox = center_Idx_ALL(oscf<0.15,:);
posbox_a = anchor_Idx_ALL(oscf>0.35,:);
pts2 = bsxfun(@plus,(posbox+1)*0.1,[-2.6,0.4,-1.5]);%plot3(pts(:,1),pts(:,2),pts(:,3),'.r');
hold on;
plot3(pts2(:,1),pts2(:,2),pts2(:,3),'.b');

a = labelCPU(:);
bb_tar_diff = readTensor_v2('/Volumes/modelnet/deepDetect/code/marvin/DSS/debug/bb_tar_diff.tensor');
bb_tar_diff_all{1} = bb_tar_diff.value(:,:,:,:,:,1);
bb_tar_diff = readTensor_v2('/Volumes/modelnet/deepDetect/code/marvin/DSS/debug/bb_tar_diff_1.tensor');
bb_tar_diff_all{2} = bb_tar_diff.value(:,:,:,:,:,1);
bb_tar_diff = zeros(26,53,53,19,6);
for gid = 0:1
    idx = find(size_group==gid);
    for i =1:size(bb_tar_diff_all{gid+1},4)
        bb_tar_diff(:,:,:,idx(i),:) = bb_tar_diff_all{gid+1}(:,:,:,i,:);
    end
end

b = reshape(bb_tar_diff,length(a),[]);
xx = b(a>0,:);

bb_loss_weights = readTensor('/Volumes/modelnet/deepDetect/code/marvin/DSS/debug/bb_loss_weights.tensor');
bb_loss_weights = bb_loss_weights.value;
b = reshape(bb_loss_weights,length(a),[]);
xx = b(a>0,:);

label_weights = readTensor_v2('/Volumes/modelnet/deepDetect/code/marvin/DSS/debug/label_weights.tensor');
label_weights_all{1} = label_weights.value(:,:,:,:,:,1);
label_weights = readTensor_v2('/Volumes/modelnet/deepDetect/code/marvin/DSS/debug/label_weights_1.tensor');
label_weights_all{2} = label_weights.value(:,:,:,:,:,1);
label_weights = zeros(26,53,53,19,2);
for gid = 0:1
    idx = find(size_group==gid);
    for i =1:size(bb_tar_diff_all{gid+1},4)
        label_weights(:,:,:,idx(i),:) = label_weights_all{gid+1}(:,:,:,i,:);
    end
end
b = reshape(label_weights,length(a),[]);
xx = b(a>0,:);