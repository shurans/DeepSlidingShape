function RPN_extract(id)

opt = dss_initPath();
seg_dir = opt.seg_dir;
load(fullfile(opt.SUNrgbd_toolbox,'Metadata/SUNRGBDMeta.mat'));

load('cls.mat');
load('anchorBox');
load('size_group_0.35.mat');
nmsthralign = 0.35;
fullinfo = 1;


NYUonly = 1; folderName = 'multi_dpcv1';            trainiter = 5000;
NYUonly = 0; folderName = 'multi_dpcv1_sunrgbd_fix';trainiter = 10000;

load('size_group_0.3.mat');
keeptop = 150;
mulit_sclae = 1;
if NYUonly
    load('./external/SUNRGBDtoolbox/traintestSUNRGBD/test_kv1NYU.mat');
    load('./external/SUNRGBDtoolbox/traintestSUNRGBD/train_kv1NYU.mat');
    allPath = [testSeq,trainSeq];
else
    load('./external/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
    allPath = [alltest,alltrain];
end

seqnames = cell(1,length(allPath));
for i =1:length(allPath)
    seqnames{i} = getSequenceName(allPath{i});
end


path2trainedmodel =['DSS/' folderName '/RPN_snapshot_' num2str(trainiter) '.marvin']; 
if  trainiter>2500
    boxLocalfolder = ['/home/shurans/deepDetectLocal/RPNresult/' folderName '_'  num2str(trainiter) '_fix/'];
    boxfolder      = ['/n/fs/modelnet/deepDetect/RPNresult/' folderName '_'  num2str(trainiter) '_fix/'];
    proposal_dir   = ['/n/fs/modelnet/deepDetect/proposal/RPN_' folderName num2str(nmsthralign) 'top' num2str(keeptop) '_'  num2str(trainiter) '_fix/'];
else
    boxLocalfolder = ['/home/shurans/deepDetectLocal/RPNresult/' folderName '_fix/'];
    boxfolder      = ['/n/fs/modelnet/deepDetect/RPNresult/' folderName '_fix/']; 
    proposal_dir   = ['/n/fs/modelnet/deepDetect/proposal/RPN_' folderName num2str(nmsthralign) 'top' num2str(keeptop) '_fix/'];
end
extractfeafile_po = ['DSS/' folderName '/DSSopt_RPN_extractfea_po.json'];
cmd =  'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64';

if mulit_sclae
    template = '%s\n mkdir %s \n./marvin test %s %s cls_score,cls_score_1,box_pred_reshape,box_pred_reshape_1 %s/cls_score,%s/cls_score_1,%s/box_pred_reshape,%s/box_pred_reshape_1 1';
    cmd = sprintf(template,cmd,boxLocalfolder, extractfeafile_po, path2trainedmodel, boxLocalfolder,boxLocalfolder,boxLocalfolder,boxLocalfolder);
    display(cmd);
else
    template = '%s\n mkdir %s \n./marvin test %s %s cls_score,box_pred_reshape %s/cls_score,%s/box_pred_reshape';
    cmd = sprintf(template,cmd,boxLocalfolder, extractfeafile_po, path2trainedmodel, boxLocalfolder,boxLocalfolder);
    display(cmd);
end
%{
cmd ='';
cmd = sprintf('%s\n rsync -aP %s   shurans@soak:%s',cmd,boxLocalfolder, boxfolder);
cmd = sprintf('%s\n rm -rf %s',cmd,boxLocalfolder);
display(cmd);
%}
%% orgnize it 
SpaceRange = [-2.6 2.6;0.4,5.6;-1.5,1];
s = 0.1;
SpaceReal  = struct('Rx',SpaceRange(1,:),'Ry',SpaceRange(2,:),'Rz',SpaceRange(3,:),'s',s);

%% loading 
if ~exist('id','var')
    imageNums = 1:length(allPath);
else
    imageNums =id:300:length(allPath);
end

%% imageNum = 1000 imageNum =393
for imageNum = imageNums
    [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
    data = SUNRGBDMeta(ind);
    tosavepath = fullfile(proposal_dir,data.sequenceName);
    if ~exist([tosavepath '.mat'],'file')
        try 
        fprintf('%s\n',tosavepath);
        if ~mulit_sclae
            scores = readTensor([boxfolder 'cls_score_' num2str(imageNum-1) '.tensor']);
            scores = scores.value(:,:,:,:,2);

            box_diff = readTensor([boxfolder 'box_pred_reshape_' num2str(imageNum-1) '.tensor']);
            box_diff = box_diff.value;
        else
            scores = readTensor([boxfolder 'cls_score_' num2str(imageNum-1) '.tensor'],false);
            scores_all{1} = scores.value(:,:,:,:,2);
            
            scores = readTensor([boxfolder 'cls_score_1_' num2str(imageNum-1) '.tensor'],false);
            scores_all{2} = scores.value(:,:,:,:,2);
            scores = zeros(26,53,53,19);
            for gid = 0:length(scores_all)-1
                idx = find(size_group==gid);
                for i =1:size(scores_all{gid+1},4)
                    scores(:,:,:,idx(i)) = scores_all{gid+1}(:,:,:,i);
                end
            end
            
            box_diff = readTensor([boxfolder 'box_pred_reshape_' num2str(imageNum-1) '.tensor'],false);
            box_diff_all{1} = box_diff.value(:,:,:,:,:,1);
            box_diff = readTensor([boxfolder 'box_pred_reshape_1_' num2str(imageNum-1) '.tensor'],false);
            box_diff_all{2} = box_diff.value(:,:,:,:,:,1);
            box_diff = single(zeros(26,53,53,19,6));
            for gid = 0:1
                idx = find(size_group==gid);
                for i =1:size(box_diff_all{gid+1},4)
                    box_diff(:,:,:,idx(i),:) = box_diff_all{gid+1}(:,:,:,i,:);
                end
            end
            
        end

        [scores_sort,idx_sort] = sort(scores(:),'descend');
        [zi,yi,xi,ai] = ind2sub(size(scores),idx_sort);
        center_Idx_ALL = [xi,yi,zi];
        % pick nonempty 
        nonempty = load([fullfile(opt.RPNdatamat_dir,[data.sequenceName]) '.mat']);
        nonemptyidx = find(ismember([center_Idx_ALL,ai],[nonempty.center_Idx_ALL,nonempty.anchor_Idx_ALL]+1,'rows'));
        
        center_Idx_ALL = center_Idx_ALL(nonemptyidx,:);
        scores_sort = scores_sort(nonemptyidx,:);
        idx_sort = idx_sort(nonemptyidx);
        ai = ai(nonemptyidx);
        
        box_pred = reshape(box_diff,numel(scores),[]);
        diff1 = box_pred(idx_sort,1:3);
        diff2 = box_pred(idx_sort,4:6);
        
       
        % convert to struct 
        ld = load(fullfile(seg_dir,data.sequenceName),'room');
        candidates3d_noreg =[];
        pickall =  [];
        for anchorId = 1:length(anchorBox)
            thisAnchorBox = find(ai == anchorId);
            center_Idx_inSpace = center_Idx_ALL(thisAnchorBox,:);
            bb_f_size = round(anchorBoxAlign(anchorId).coeffs/SpaceReal.s);
            %bbs_3d_f = [center_Idx_inSpace repmat(bb_f_size,[size(center_Idx_inSpace,1),1])];
            bbs_3d_f = [bsxfun(@minus,center_Idx_inSpace,0.5*(bb_f_size-mod(bb_f_size,2))),repmat(bb_f_size,[size(center_Idx_inSpace,1),1])];
            bbw =bbf2w(bbs_3d_f,SpaceReal); 
            pick = nmsMe_3d([bbw(:,1:3),bbw(:,1:3)+bbw(:,4:6),scores_sort(thisAnchorBox)], nmsthralign);
            pick = sort(pick);
            pick = pick(1:min(length(pick),keeptop));
            pickall = [pickall;thisAnchorBox(pick)];
            bbw = bbw(pick,:);
            candidates3d_noreg = [candidates3d_noreg;transform2tightBB(bbw,anchorBox(anchorId),ld.room.Rot)];
        end

        % add regression  
        candidates3d_noreg = processbox(candidates3d_noreg);
        diff1 = diff1(pickall,:);
        diff2 = diff2(pickall,:);
        scores_sort = scores_sort(pickall,:);

        centroid = reshape([candidates3d_noreg.centroid],[3,length(candidates3d_noreg)])';
        coeffs = reshape([candidates3d_noreg.coeffs],[3,length(candidates3d_noreg)])';
        centroid = diff1.*coeffs+centroid;
        coeffs = exp(diff2).*coeffs;
        candidates3d = candidates3d_noreg;
        for i =1:length(candidates3d_noreg)
            candidates3d(i).centroid = centroid(i,:);
            candidates3d(i).coeffs  =  coeffs(i,:);
            candidates3d(i).conf = scores_sort(i);
            candidates3d_noreg(i).conf = scores_sort(i);
        end

        [~,scoresortidx] = sort([candidates3d.conf],'descend');
        candidates3d_noreg = candidates3d_noreg(scoresortidx);
        candidates3d = candidates3d(scoresortidx);
        %% caculate overlap
        gt_bb = data.groundtruth3DBB;
        oscfM_r = bb3dOverlapCloseForm(candidates3d,gt_bb');
        
        
        
        if fullinfo
            %% get the 2D boxes for each 3d box
            [rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
            for j =1:length(candidates3d)
                 [box_tight,~,box_proj] = get2Dbbwithdepth(candidates3d(j),imsize,points3d,depthInpaint,data);
                 candidates3d(j).box2d_tight = box_tight;
                 candidates3d(j).box2d_proj = box_proj;
            end 
            removeid = [];
            for i =1:length(candidates3d)
                if isempty(candidates3d(i).box2d_tight)
                    removeid = [removeid,i];
                end
            end
            candidates3d(removeid) =[];
            if ~isempty(oscfM_r)
                oscfM_r(removeid,:) =[];
            end
            
            %% give candidates3d labels 
            if ~isempty(gt_bb)
                candidates3d = processbox(candidates3d); 
                gt_bb = processbox(gt_bb);
                [oscf,matchgt] = max(oscfM_r,[],2);
               
                for bi = 1:length(candidates3d)
                    candidates3d(bi).iou = oscf(bi);
                    if candidates3d(bi).iou >0
                        candidates3d(bi).o = atan2(gt_bb(matchgt(bi)).orientation(1),gt_bb(matchgt(bi)).orientation(2))/pi*180;
                        candidates3d(bi).classname = gt_bb(matchgt(bi)).classname;
                        candidates3d(bi).diff = get_regdiff_detect(candidates3d(bi),gt_bb(matchgt(bi)));
                        
                    else
                        candidates3d(bi).classname = 'negative';
                        candidates3d(bi).diff = [];
                    end
                end
                % get 2d gt match 
                box2d_proj = reshape([candidates3d.box2d_proj],4,[]);
                oscfM = bb2dOverlap(box2d_proj',data.groundtruth2DBB_full'); 
                [oscf,matchgt] = max(oscfM,[],2);
                for bi = 1:length(candidates3d)
                    if candidates3d(bi).iou >0&&oscf(bi)>0.1
                       candidates3d(bi).diff_2dp = data.groundtruth2DBB_full(matchgt(bi)).gtBb2D - candidates3d(bi).box2d_proj;
                    else
                       candidates3d(bi).diff_2dp =[];
                    end
                end
                if ~isempty(data.groundtruth2DBB_tight)
                    [has2d,map] = ismember([1:length(data.groundtruth2DBB_full)],[data.groundtruth2DBB_tight.box3didx]);
                    oscfM = zeros(size(oscfM,1),size(oscfM,2));
                    box2d_tight = reshape([candidates3d.box2d_tight],4,[]);
                    oscfM_x = bb2dOverlap(box2d_tight',data.groundtruth2DBB_tight'); 
                    oscfM(:,find(has2d)) = oscfM_x(:,map(has2d));
                    [oscf,matchgt] = max(oscfM,[],2);
               
                    for bi = 1:length(candidates3d)
                        if candidates3d(bi).iou >0&&oscf(bi)>0.1
                           candidates3d(bi).diff_2dt = data.groundtruth2DBB_tight(matchgt(bi)).gtBb2D - candidates3d(bi).box2d_tight; 
                        else
                           candidates3d(bi).diff_2dt =[];
                        end
                    end
                else
                    for bi = 1:length(candidates3d)
                        candidates3d(bi).diff_2dt =[];
                    end
                end
            else
                for bi = 1:length(candidates3d)
                    candidates3d(bi).iou = 0;
                    candidates3d(bi).classname = 'negative';
                    candidates3d(bi).diff = [];
                    candidates3d(bi).diff_2dp =[];
                    candidates3d(bi).diff_2dt =[];
                end
            end
        end
        %% save it 
        oscfM_r = sparse(double(oscfM_r));
        if ~isempty(gt_bb)
            gtnames = {gt_bb.classname};
        else
            gtnames =[];
        end
        ind = find(tosavepath =='/');mkdir(tosavepath(1:ind(end)));
        save([tosavepath '.mat'],'candidates3d','oscfM_r','gtnames');
        end
        
        % visulize it 
        if 0
            clf 
            imshow(data.rgbpath)
            for bi = 1:length(candidates3d)
                hold on;
                rectangle('Position',  data.groundtruth2DBB_tight(matchgt(bi)).gtBb2D,'edgecolor','r');
                rectangle('Position',  data.groundtruth2DBB_full(matchgt(bi)).gtBb2D,'edgecolor','g');
                rectangle('Position',  candidates3d(bi).box2d_tight,'edgecolor','b');
                rectangle('Position',  candidates3d(bi).box2d_proj,'edgecolor','c');
                rectangle('Position',  candidates3d(bi).box2d_tight+candidates3d(bi).diff_2dt,'edgecolor','y');
                rectangle('Position',  candidates3d(bi).box2d_proj+candidates3d(bi).diff_2dp,'edgecolor','m');
                pause;
            end
            clf
            [GTmaxOverlap,bestCand] = max(oscfM_r);
            [rgb,points3d,imgZ]=read3dPoints(data);
            vis_point_cloud(points3d,rgb,10,10000);
            hold on;
            for bid = 1:length(candidates3d)
                %vis_cube(candidates3d_noreg(bid),'b');
                vis_cube(candidates3d(bid),'r');
                pause;
            end

            figure,
            clf
            vis_point_cloud(points3d,rgb,10,10000);
            hold on;
            for bid = 1:length(gt_bb)
                vis_cube(gt_bb(bid),'g');
                %vis_cube(candidates3d_noreg(bestCand(bid)),'b');
                vis_cube(candidates3d(bestCand(bid)),'r');
                pause;
            end
        end
    else
        fprintf('skiping %s\n',tosavepath);
    end
end

if fullinfo&&~exist('id','var')
   [means,stds,means_2d,stds_2d]=mean_std_boxdiff_det(proposal_dir,1);
end
end

function saveinparfor(tosavepath,candidates3d_noreg, candidates3d_reg ,oscfM)
         save([tosavepath '.mat'],'candidates3d_noreg','candidates3d_reg','oscfM');   
end