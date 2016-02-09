function mRecall(boxType,evalType,folderName,BoxperImage)
% evaluate recall for all anchor, for all nonempty anchor 
% boxType = {'seg','nonemptyAnchor','RPN','RPNreg','from2D'}
% evalType = {'bb2d_tight','bb3d','box2d_proj'}
% mRecall('RPNreg','bb3d','RPN_multi_dpcv1_sunrgbd_fix0.35top150_10000_fix', 2000)
% mRecall('seg','bb3d','sunrgbd', 2000)
cd /n/fs/modelnet/deepDetect/code/detector_3d/
dss_initPath;
addpath /n/fs/modelnet/SUN3DV2/prepareGT/
setup_benchmark;
train = 0;
NYUonly = 1;

if NYUonly
    if train
        load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat')
        allPath = trainSeq;
    else
        load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
        allPath = testSeq;
    end
else
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/allsplit.mat')
    allPath = alltest;
end

for i =1:length(allPath)
    seqnames{i} = getSequenceName(allPath{i});
end
SpaceRange = [-2.6 2.6;0.4,5.6;-1.5,1];
s = 0.1;
SpaceReal  = struct('Rx',SpaceRange(1,:),'Ry',SpaceRange(2,:),'Rz',SpaceRange(3,:),'s',s);

categoryids =[];
imageids    =[];
overlap     =[];
numofbox = zeros(1,length(seqnames));
load('/n/fs/modelnet/SUN3DV2/prepareGT/Metadata/SUNRGBDMeta_tight_Yaw.mat');
load(['/n/fs/modelnet/SUN3DV2/prepareGT/cls.mat']);
load('anchorBox');
seg_dir ='/n/fs/modelnet/deepDetect/seg/';

for imageNum =1:length(seqnames)
    [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
    data = SUNRGBDMeta(ind);
    gt_bb = data.groundtruth3DBB;
    if ~isempty(gt_bb)
        [carebox,classid] = ismember({gt_bb.classname},cls);
        gt_bb = gt_bb(find(carebox));
    end
    
    if ~isempty(gt_bb)
        if strcmp(boxType,'nonemptyAnchor')
            load([fullfile('/n/fs/modelnet/deepDetect/RPNdata_mulit/',[data.sequenceName]) '.mat']);
            oscfM = full(oscfM);
            %{
               load(fullfile(seg_dir,data.sequenceName),'room');

               candidates3d =[];
               for anchorId = 1:length(anchorBox)
                   center_Idx_inSpace = center_Idx_ALL(anchor_Idx_ALL == anchorId,:);
                   bb_f_size = round(anchorBoxAlign(anchorId).coeffs/SpaceReal.s);
                   %bbs_3d_f = [center_Idx_inSpace+1 repmat(bb_f_size,[size(center_Idx_inSpace,1),1])];
                   bbs_3d_f = [bsxfun(@minus,center_Idx_inSpace+1,0.5*bb_f_size),repmat(bb_f_size,[size(center_Idx_inSpace,1),1])];
                   bbw =bbf2w(bbs_3d_f,SpaceReal); 
                   candidates3d = [candidates3d;transform2tightBB(bbw,anchorBox(anchorId),room.Rot)];
               end
            %}
        end
        
        if  strcmp(boxType,'from2D')
            load([fullfile('/n/fs/modelnet/deepDetect/proposal/pofrom2dbox/',data.sequenceName) '.mat.mat']) 
            oscfM = full(oscfM_r);
            [carebox] = find(ismember(gtnames,cls));
            oscfM = oscfM(:,carebox);
        end
        
        if strcmp(boxType,'seg')
           load([fullfile('/n/fs/modelnet/deepDetect/proposal/rgbd_tight/',data.sequenceName) '.mat']) 
           oscfM = bb3dOverlapCloseForm(candidates3d,gt_bb');
        end
        
        if strcmp(boxType,'RPN')
           load([fullfile(['/n/fs/modelnet/deepDetect/proposal/' folderName '/'],data.sequenceName) '.mat']) 
           oscfM = full(oscfM_n);
           [carebox] = find(ismember(gtnames,cls));
           oscfM = oscfM(:,carebox);
           oscfM = oscfM(1:min(size(oscfM,1),BoxperImage),:);
        end
        
        if strcmp(boxType,'RPNreg')
           load([fullfile(['/n/fs/modelnet/deepDetect/proposal/' folderName '/'],data.sequenceName) '.mat']) 
           if strcmp(evalType,'bb2d_tight')
              box2d_tight = reshape([candidates3d.box2d_tight],4,[])';
              gtbb = data.groundtruth2DBB_tight([data.groundtruth2DBB_tight.box3didx]>0);
              oscfM = bb2dOverlap(box2d_tight,gtbb');
           elseif strcmp(evalType,'box2d_proj')
              box2d_proj = reshape([candidates3d.box2d_proj],4,[])';
              gtbb = data.groundtruth2DBB_full;
              oscfM = bb2dOverlap(box2d_proj,gtbb'); 
           else
               [carebox] = find(ismember(gtnames,cls));
               oscfM = full(oscfM_r);
               oscfM = oscfM(:,carebox);
           end
           oscfM = oscfM(1:min(size(oscfM,1),BoxperImage),:);
        end
        
        
        numofbox(imageNum) = min(length(candidates3d),BoxperImage);
        [GTmaxOverlap,bestCand] = max(oscfM);
        categoryids = [categoryids,classid(carebox)];
        imageids = [imageids,imageNum*ones(1,length(gt_bb))];
        overlap = [overlap,GTmaxOverlap];        
    end
end 

save(['/n/fs/modelnet/deepDetect/code/detector_3d/eval/RPNresult/' boxType '_' folderName '_' num2str(BoxperImage) '_mrecall.mat'],'imageids','categoryids','overlap','numofbox');

if strcmp(evalType,'bb2d_tight')||strcmp(evalType,'box2d_proj')
    thr = 0.5;
else
    thr = 0.25;
end

for cid = 1:length(cls)
    oscls = overlap(categoryids == cid);
    fprintf('%f \n',mean(oscls>thr)*100);
end
fprintf('%f \n',mean(overlap>thr)*100);
fprintf('%f \n',mean(overlap));
fprintf('%f \n',mean(numofbox(numofbox>0)));

%%
allresult = {'from2D__Inf_mrecall','seg_Inf_0.35mrecall','RPNreg_fullbox_f102_2000_0.35mrecall',...
             'RPNreg_RPN_multi_dpcv1_d0.35top150_5000_fix_2000_mrecall','RPNreg_RPN_multi_dpcv10.35top150_5000_fix_2000_mrecall','nonemptyAnchor_Inf_0.35mrecall'};
thres = [0.1:0.05:1];
result_curve = zeros(length(allresult),length(thres));
for i =1:length(allresult)
    load(fullfile('/n/fs/modelnet/deepDetect/code/detector_3d/eval/RPNresult/',[allresult{i} '.mat']));
    for j = 1:length(thres)
        result_curve(i,j) = mean(overlap>thres(j))*100;
    end
    
end
clf
plot(thres,result_curve(1:end-1,:)','LineWidth',3);
hold on;
plot(thres,result_curve(end,:)','--','LineWidth',3);
h=legend({'2D To 3D','3D Selective Search','RPN Single','RPN Multi','RPN Multi + Color','All Anchors'})
set(h,'FontSize',20)



%{
[~,ind]=ismember(seqnames,{SUNRGBDMeta.sequenceName});
groundtruth3DBB = [SUNRGBDMeta(ind).groundtruth3DBB];
groundtruth3DBB = groundtruth3DBB(ismember({groundtruth3DBB.classname},cls));
missgt = groundtruth3DBB(categoryids == 10 & overlap<0.35);

a.imageids(w.overlap>0.25&a.overlap<0.25)
missgt = groundtruth3DBB(a.categoryids==17&a.overlap<0.35);


figure
for i =1:length(missgt)
    hold on;
    vis_cube(missgt(i),'g');
    axis equal;
    pause;
end
%}