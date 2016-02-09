function boxfile = dss_prepare_synlist(sysn_dataroot,proposal_dir,outpath,cls,BoxperImage)
% train_or_test = 'train';
% proposal = 1;
NYUonly = 1;
MAXRenderPerimage = 10;
if ~exist('sysn_dataroot','var')||~exist(sysn_dataroot,'dir')
    sysn_dataroot = '/n/fs/sunhome/results_nyu/';
end
if ~exist('proposal_dir','var')
    proposal_dir = ['/n/fs/modelnet/deepDetect/proposal/RPN_multi_lv10.30.35top150_5000'];
end
if ~exist('outpath','var')
    outpath = proposal_dir;
end

if ~exist('BoxperImage','var')
    BoxperImage = 2000;
end

if ~exist('cls','var')
    load(['/n/fs/modelnet/SUN3DV2/prepareGT/cls.mat']);
end

if NYUonly
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat')
    alltrain = trainSeq;
    alltest = testSeq;
    filename = 'syn_NYU_results_nyu_test';
else
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/allsplit.mat')
    filename = 'syn_SUNrgbd';
end
allpath = [alltrain];
seqnames = cell(1,length(allpath));
for i =1:length(allpath)
    seqnames{i} = getSequenceName(allpath{i});
end

writeTarget = 1;
filename = [filename '_diff'];
mean_std = load(fullfile(proposal_dir,'mean_std'),'means','stds');


SUNRGBDtoolboxdir = '/n/fs/modelnet/SUN3DV2/prepareGT/';
boxfile = [outpath '/' filename '.list'];
%%
if ~exist(boxfile,'file')
    fprintf('out: %s\n',boxfile);
    load([SUNRGBDtoolboxdir '/Metadata/' 'SUNRGBDMeta_tight_Yaw.mat']);
    fid = fopen(boxfile,'wb');
    totalnumofbox =0;
    for imageNum = 1:length(seqnames)
        fprintf('%d image: %s \n', imageNum,seqnames{imageNum})
        [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
        data = SUNRGBDMeta(ind);
        if ~isempty(data.groundtruth3DBB)
            % if gt is not empty for all syn images write to list
            allsysnfile = dir([fullfile(sysn_dataroot,data.sequenceName) '/*.tensor']);
            % load boxes
            candidatefilename = [fullfile(proposal_dir,data.sequenceName) '.mat'];
            ld = load([fullfile(proposal_dir,data.sequenceName) '.mat']);
            boxes = ld.candidates3d;
            if BoxperImage>0
                boxes = boxes(1:min(BoxperImage,length(boxes)));
            end
            
            % 
            for syn_id =1:min(MAXRenderPerimage,length(allsysnfile))
               
                tensor_filename = fullfile(data.sequenceName,allsysnfile(syn_id).name);
                %{
                T = readTensor(fullfile(sysn_dataroot,tensor_filename));
                depth_render = T(1).value';
                if max(depth_render(:))<1.5
                    continue;
                end
                %}   
                load(strrep(fullfile(sysn_dataroot,tensor_filename),'tensor','mat'));
                if isempty(obj_metadata)
                    mecls ={'negative'};
                else
                    mecls = [{obj_metadata.class},'negative'] ;
                    mecls = mecls(~cellfun('isempty',mecls)) ;
                end
                notexistgt = find(~ismember({boxes.classname},mecls));
                for k =1:length(notexistgt)
                    boxes(notexistgt(k)).classname = 'negative';
                end
                %{
                    T = readTensor(fullfile(sysn_dataroot,tensor_filename),true);
                    depth_render = T(1).value;
                    camRT = T(2).value';
                    clear gtbb;
                    cnt = 1;
                    for bid= 1:length(obj_metadata)
                        if ~isempty(obj_metadata(bid).rMatrix)
                            gtbb(cnt).basis = eye(3,3);
                            gtbb(cnt).basis(1:2,1:2) = obj_metadata(bid).rMatrix;
                            gtbb(cnt).centroid = obj_metadata(bid).centroid;
                            gtbb(cnt).coeffs = obj_metadata(bid).coeffs;
                            cnt = cnt+1;
                        end
                    end

                    [rgb,points3d]=read_3d_pts_general(depth_render,data.K,size(depth_render),[]);
                    points3d = (data.Rtilt*points3d')';
                    figure,
                    vis_point_cloud(points3d,[],10,10000);
                    hold on;
                    for i =1:length(gtbb)
                        vis_cube(gtbb(i),'r')
                    end

                    for i =1:length(gtbb)
                        vis_cube(data.groundtruth3DBB(i),'b')
                    end
                %}
                fwrite(fid, uint32(length(tensor_filename)), 'uint32');
                % TODO : load tensor and get the RT to apply to the R
                % TODO : for the category not exist in the shapenet don't
                % add it as postive 
                fwrite(fid, tensor_filename, 'char*1');
                R = data.Rtilt';
                fwrite(fid, single(R), 'single'); 
                K = data.K';
                fwrite(fid, single(K), 'single'); 
                imsize = size(imread(data.rgbpath));
                fwrite(fid, uint32(imsize(1)), 'uint32'); 
                fwrite(fid, uint32(imsize(2)), 'uint32');
                
                numbox = length(boxes);
                fwrite(fid, uint32(numbox), 'uint32');
                for bi =1:numbox
                    [~,classid]= ismember(boxes(bi).classname,cls);
                    overlap =boxes(bi).iou;
                    if classid==0
                       overlap =0;
                    end
                    if overlap<0.25
                       classid =0;
                    end
                    fwrite(fid,uint32(classid), 'uint32');
                    B = boxes(bi).basis';
                    fwrite(fid,single(B), 'single');
                    fwrite(fid,single(boxes(bi).centroid),'single');
                    fwrite(fid,single(boxes(bi).coeffs),'single');
                    totalnumofbox = totalnumofbox+1;

                    if writeTarget
                       if  classid>0
                           fwrite(fid,uint8(1), 'uint8'); % uint8
                           diff = boxes(bi).diff;
                           diff = diff - mean_std.means(classid,:);
                           diff = diff./mean_std.stds(classid,:);
                           fwrite(fid,single(diff(:)), 'single');
                       else
                           boxes(bi).gtbb =[];
                           fwrite(fid,uint8(0), 'uint8');
                       end

                    end
                end
            end
        end
    end
end

if 0
    imagecount = 1;
    fid = fopen('/n/fs/modelnet/deepDetect/proposal/RPN_multi_lv10.30.35top150_5000//syn_NYU_diff_back.list');

    while 1 
        
        len = fread(fid , 1, 'int32');	
        name = fread(fid , len,  'char*1');	
        name = [char(name')];
        display(name)

        R = fread(fid , 9, 'single');	
        K = fread(fid , 9, 'single');	
        h = fread(fid , 1, 'uint32');	
        w = fread(fid , 1, 'uint32');	
        len = fread(fid , 1, 'uint32');
        clear box;
        for i =1:len
            box(i).category = fread(fid , 1, 'uint32');
            box(i).basis = reshape(fread(fid , 9, 'single'),3,3);	
            box(i).centroid = fread(fid , 3, 'single');	
            box(i).coeffs = fread(fid , 3, 'single');	

            hasTarget = fread(fid , 1, 'uint8');
            if hasTarget>0
               box(i).diff = fread(fid , 6, 'single');	
            end

        end
        imagecount = imagecount+1;
    end
end
            
        
end
