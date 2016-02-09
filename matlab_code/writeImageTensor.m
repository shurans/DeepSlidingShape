%{
load('/n/fs/modelnet/SUN3DV2/prepareGT/Metadata/SUNRGBDMeta_tight_Yaw.mat');
for i =1:length(SUNRGBDMeta)
    data = SUNRGBDMeta(i);
    image = imread(data.rgbpath);
    images_tensor(1).name = 'image';
    images_tensor(1).value = uint8(image);
    images_tensor(1).type = 'uint8';
    images_tensor(1).sizeof = 1;
    images_tensor(1).dim = 4;
    out_file = ['/n/fs/modelnet/deepDetect/image/' data.sequenceName '.tensor'];
    ind = find(out_file =='/');
    mkdir(out_file(1:ind(end)));
    writeTensor(out_file, images_tensor);
end
%}
%{
for i =2:length(SUNRGBDMeta)
    data = SUNRGBDMeta(i);
    out_file = ['/n/fs/modelnet/deepDetect/hha/' data.sequenceName '.tensor'];
    scr = ['/n/fs/modelnet/deepDetect/ucm/' data.sequenceName '.tensor'];
    if exist(scr,'file')
       ind = find(out_file =='/');
       mkdir(out_file(1:ind(end)));
       movefile(scr,out_file);
    end
end
%}

clear all
load('/n/fs/modelnet/SUN3DV2/prepareGT/Metadata/SUNRGBDMeta_tight_Yaw.mat');
for i = 1:length(SUNRGBDMeta)
    data = SUNRGBDMeta(i);
    
    %in_file = ['/n/fs/modelnet/deepDetect/image_org/' data.sequenceName '.tensor'];
    %out_file = ['/n/fs/modelnet/deepDetect/image/' data.sequenceName '.tensor'];
    in_file = ['/n/fs/modelnet/deepDetect/hha_org/' data.sequenceName '.tensor'];
    out_file = ['/n/fs/modelnet/deepDetect/hha/' data.sequenceName '.tensor'];
   
    display(in_file);
    
    T = readTensor(in_file, 'true');
    im = zeros(688,904,3);
    scale = min(size(im,1)/size(T.value,1),size(im,2)/size(T.value,2));
    im_recale = imresize(T.value,scale,'bilinear');
    im(1:size(im_recale,1),1:size(im_recale,2),:) = im_recale;
    T.value = single(im);
    
    T.value = T.value(:,:,[3,2,1]);
    T.value(:,:,1) = T.value(:,:,1) - 102.9801;
    T.value(:,:,2) = T.value(:,:,2) - 115.9465;
    T.value(:,:,3) = T.value(:,:,3) - 122.7717;
    T.type = 'half';
    T.sizeof = 2;
    
    ind = find(out_file =='/'); mkdir(out_file(1:ind(end)));
    writeTensor(out_file, T);
end
%}

%{
load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat')
alltrain = trainSeq;
alltest = testSeq;
allPath = [alltest,alltrain];
seqnames = cell(1,length(allPath));
for bi =1:length(allPath)
    seqnames{bi} = getSequenceName(allPath{bi});
end


for imageNum = 1:length(seqnames)
    [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
    data = SUNRGBDMeta(ind);
   
    out_file = ['/n/fs/modelnet/deepDetect/image_crop/' data.sequenceName '.tensor'];
    img_id = str2double(data.sequenceName(length('SUNRGBD/kv1/NYUdata/NYU')+1:end));
    imname = ['img_' num2str(img_id+5000) '.png'];
    
    T.name = 'image';
    im = imread(['/n/fs/modelnet/Gupta/CrossModal/fast-rcnn/data/nyud2/data/images/' imname]);
    im = imresize(im,688/size(im,1),'bilinear');
    im = single(im(:,:,[3 2 1]));
    im(:,:,1) = im(:,:,1) - 102.9801;
    im(:,:,2) = im(:,:,2) - 115.9465;
    im(:,:,3) = im(:,:,3) - 122.7717;

    T.value  = im;
    T.type   = 'half';
    T.sizeof = 2;
    ind = find(out_file =='/'); mkdir(out_file(1:ind(end)));
    writeTensor(out_file, T);
end
%}
%{
for imageNum = 1:length(seqnames)
    [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
    data = SUNRGBDMeta(ind);
   
    out_file = ['/n/fs/modelnet/deepDetect/hha_crop/' data.sequenceName '.tensor'];
    img_id = str2double(data.sequenceName(length('SUNRGBD/kv1/NYUdata/NYU')+1:end));
    imname = ['img_' num2str(img_id+5000) '.png'];
    
    T.name = 'hha';
    im = imread(['/n/fs/modelnet/Gupta/CrossModal/fast-rcnn/data/nyud2/data/hha/' imname]);
    im = imresize(im,688/size(im,1),'bilinear');
    im = single(im(:,:,[3 2 1]));
    im(:,:,1) = im(:,:,1) - 102.9801;
    im(:,:,2) = im(:,:,2) - 115.9465;
    im(:,:,3) = im(:,:,3) - 122.7717;
    T.value = im;
    T.type = 'half';
    T.sizeof = 2;
    ind = find(out_file =='/'); mkdir(out_file(1:ind(end)));
    writeTensor(out_file, T);
end
%}

