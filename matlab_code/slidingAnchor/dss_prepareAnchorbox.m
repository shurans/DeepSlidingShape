function dss_prepareAnchorbox(id)
% cd /n/fs/modelnet/deepDetect/Release/code/matlab_code/slidingAnchor
% /n/fs/vision/ionicNew/starter.sh dss_prepareAnchorbox 2000mb 165:00:00 1 300 1 

cd ..
opt = dss_initPath();
tosave_dir = opt.RPNdata_dir;
seg_dir    = opt.seg_dir;
NYUonly =1;
if NYUonly
    load('./external/SUNRGBDtoolbox/traintestSUNRGBD/test_kv1NYU.mat');
    load('./external/SUNRGBDtoolbox/traintestSUNRGBD/train_kv1NYU.mat');
    allPath = [testSeq,trainSeq];
else
   load('./external/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
   allPath = [alltest,alltrain];
end
load('anchorBox'); 
load('cls.mat');

load('./external/SUNRGBDtoolbox/Metadata/SUNRGBDMeta_tight_Yaw.mat');


s = 0.1;
margin = 0.4;
margin_f = margin/s;
SpaceRange = [-2.6 2.6;0.4,5.6;-1.5,1];
outputSize = [SpaceRange(:,2)-SpaceRange(:,1)]'/s+1;%[19, 53, 53, 26];
SpaceRangefull = [SpaceRange(:,1)-margin,SpaceRange(:,2)+margin];

Space  = struct('Rx',SpaceRangefull(1,:),'Ry',SpaceRangefull(2,:),'Rz',SpaceRangefull(3,:),'s',s);
SpaceReal  = struct('Rx',SpaceRange(1,:),'Ry',SpaceRange(2,:),'Rz',SpaceRange(3,:),'s',s);
bincellDim = uint16([SpaceRangefull(:,2)-SpaceRangefull(:,1)]'/s+1);

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
    tosavepath = fullfile(tosave_dir,[data.sequenceName]);
    
    if ~(exist([tosavepath '.bin'],'file'))
        fprintf('computing %s\n',tosavepath);
        [rgb,points3d,imgZ]=read3dPoints(data);
        
        load(fullfile(seg_dir,data.sequenceName),'room','imageSeg');
        [minZ,maxZ,minX,maxX,minY,maxY,wall_floor] = getRoom(points3d,imageSeg,room.Rot,0);

        inside = points3d(:,1)>SpaceRangefull(1,1)&points3d(:,1)<SpaceRangefull(1,2)&...
                 points3d(:,2)>SpaceRangefull(2,1)&points3d(:,2)<SpaceRangefull(2,2)&...
                 points3d(:,3)>SpaceRangefull(3,1)&points3d(:,3)<SpaceRangefull(3,2);

        points3d = points3d(inside&(wall_floor(:)<1),:);
        xind   = floor((points3d(:,1)-SpaceRangefull(1,1))/Space.s);
        yind   = floor((points3d(:,2)-SpaceRangefull(2,1))/Space.s);
        zind   = floor((points3d(:,3)-SpaceRangefull(3,1))/Space.s);
        cellIdx= [xind+1,yind+1,zind+1];
        
               
        [pointCountIntegral,pointCount] =getIntegralPtCount(bincellDim,cellIdx);
        
        center_Idx_ALL =[];
        anchor_Idx_ALL =[];
        candidates3d =[];
        for anchorId = 1:length(anchorBox)
            bb_f_size = round(anchorBoxAlign(anchorId).coeffs/Space.s);
            volumn = prod(anchorBox(anchorId).coeffs);
            nonemptyThrehold = volumn*5000;
            [EmptyBox, countBox]= EmptyBoxFlag(pointCountIntegral,bb_f_size,nonemptyThrehold);
            [indexes1, indexes2, indexes3] = ind2sub(size(EmptyBox),find(~EmptyBox(:)));
            %% only pick the  centered in Space
            center_f = bsxfun(@plus,[indexes1, indexes2, indexes3],0.5*bb_f_size);
            inSpace  = center_f(:,1)>margin_f+0.5&center_f(:,2)>margin_f+0.5&center_f(:,3)>margin_f+0.5&...
                       center_f(:,1)<bincellDim(1)-margin_f+1&...
                       center_f(:,2)<bincellDim(2)-margin_f+1&...
                       center_f(:,3)<bincellDim(3)-margin_f+1;

            xyz_Idx_inSpace = floor(center_f(inSpace,:)) - margin_f - 1;  % -1 for c++
            center_Idx_ALL = [center_Idx_ALL;xyz_Idx_inSpace];
            anchor_Idx_ALL = [anchor_Idx_ALL;(anchorId-1)*ones(size(xyz_Idx_inSpace,1),1)];
            
            
            %% debug
            %bbs_3d_f = [center_Idx_inSpace+1 repmat(bb_f_size,[size(center_Idx_inSpace,1),1])];
            %bbs_3d_f = [bsxfun(@minus,center_Idx_inSpace+1,0.5*bb_f_size),repmat(bb_f_size,[size(center_Idx_inSpace,1),1])];
            bbs_3d_f = [bsxfun(@minus,xyz_Idx_inSpace+1,0.5*(bb_f_size-mod(bb_f_size,2))),repmat(bb_f_size,[size(xyz_Idx_inSpace,1),1])];
            bbw =bbf2w(bbs_3d_f,SpaceReal); 
            candidates3dthis = transform2tightBB(bbw,anchorBox(anchorId),room.Rot);
            candidates3dthis = processbox(candidates3dthis);
            candidates3d = [candidates3d;candidates3dthis];
            %{
            clf 
            vis_point_cloud(points3d,rgb,10,10000);
             hold on;
             for bid = 1:length(gt_bb)
                 vis_cube(gt_bb(bid),'r');
             end
             for bid = 1:10:length(candidates3d)
                 if (oscf(bid)>0.35)
                     vis_cube(candidates3d(bid),'b');
                     pause;
                 end
             end
            %}
        end     
        %%
        gt_bb = data.groundtruth3DBB;
        if ~isempty(gt_bb)
           gt_bb = gt_bb(find(ismember({gt_bb.classname},cls)));
        end
        if isempty(gt_bb)
            oscf = zeros(size(candidates3d,1),1);
            diff = zeros(6,size(candidates3d,1));
            oscfM =[];
        else
            oscfM=bb3dOverlapCloseForm(candidates3d,gt_bb');
            [GTmaxOverlap,bestCand] = max(oscfM);
            [oscf,matchgt] = max(oscfM,[],2);
            missgt = find(GTmaxOverlap<=0.35&GTmaxOverlap>0.25);
            if ~isempty(missgt)
                oscf(bestCand(missgt)) = 0.36;
            end
        
            gt_bb = processbox(gt_bb);
            
            diff1 = reshape([gt_bb(matchgt).centroid] - [candidates3d.centroid],[3,length(candidates3d)]);
            diff1 = diff1./reshape([candidates3d.coeffs],[3,length(candidates3d)]);
            
            diff2 = zeros(3,length(candidates3d));
            for bi = 1:length(candidates3d)
                dotproduct = sum(repmat(candidates3d(bi).basis(1,:),3,1).*gt_bb(matchgt(bi)).basis,2);
                [~,ind] = sort(dotproduct(1:2),'descend');
                diff2(1:3,bi) = log(gt_bb(matchgt(bi)).coeffs([ind;3])./candidates3d(bi).coeffs);
            end
            %diff2 = reshape(log([gt_bb(matchgt).coeffs]./[candidates3d.coeffs]),[3,length(candidates3d)]);
            diff = [diff1;diff2];
            diff(:,find(oscf<0.01)) =0;
            oscfM(oscfM<0.01) = 0;
            oscfM = sparse(double(oscfM));
        end
        %{
        if 0
            centroid = reshape([candidates3d.centroid],[3,length(candidates3d)])';
            coeffs = reshape([candidates3d.coeffs],[3,length(candidates3d)])';
            centroid = diff1'.*coeffs+centroid;
            coeffs = exp(diff2').*coeffs;
            regressbox = candidates3d;
            clf
            vis_point_cloud(points3d,rgb,10,10000);
            hold on;
            for i =1:length(candidates3d)
                regressbox(i).centroid = centroid(i,:);
                regressbox(i).coeffs  =  coeffs(i,:);
                if oscf(i)>0.25
                    vis_cube(candidates3d(i),'r');hold on;
                    vis_cube(regressbox(i),'b');
                    vis_cube(gt_bb(matchgt(i)),'g');
                    pause;
                end
            end
        end
        %}
        % save matlab 
        center_Idx_ALL = single(center_Idx_ALL);
        anchor_Idx_ALL = single(anchor_Idx_ALL);
        
        ind = find(tosavepath =='/');
        mkdir(tosavepath(1:ind(end)));
     
        save([tosavepath '.mat'], 'center_Idx_ALL','anchor_Idx_ALL','oscfM','diff');
        % save binary linear index with anchor_Idx_ALL center_Idx_ALL+ oscf
        %{
        linearInd = anchor_Idx_ALL*outputSize(1)*outputSize(2)*outputSize(3)+...
                    center_Idx_ALL(:,1)*outputSize(2)*outputSize(3)+...
                    center_Idx_ALL(:,2)*outputSize(3)+...
                    center_Idx_ALL(:,3);
        %}
        fid = fopen([tosavepath '.bin'],'wb');
        fwrite(fid,uint32(length(anchor_Idx_ALL)),'uint32');
        %fwrite(fid,uint32(linearInd),'uint32');
        fwrite(fid,uint8(anchor_Idx_ALL),'uint8');
        fwrite(fid,uint8(center_Idx_ALL(:,1)),'uint8');
        fwrite(fid,uint8(center_Idx_ALL(:,2)),'uint8');
        fwrite(fid,uint8(center_Idx_ALL(:,3)),'uint8');
        fwrite(fid,single(oscf),'single');
        fwrite(fid,single(diff),'single');
    else
        if exist([tosavepath '.mat'],'file')
            tomovepath = strrep(tosavepath,'RPNdata_mulit','RPNdata_mulitmat');
            ind = find(tomovepath =='/');
            mkdir(tomovepath(1:ind(end)));
            movefile([tosavepath '.mat'],[tomovepath '.mat'])
            fprintf('moving %s to %s\n',[tosavepath '.mat'],[tomovepath '.mat']);
        end
    end
end

end
%{
if 0
    figure,
    clf
    vis_point_cloud(points3d,rgb,10,10000);
    hold on;
    for bid = 1:length(gt_bb)
         vis_cube(gt_bb(bid),'g');
         vis_cube(candidates3d(bestCand(bid)),'r');
         pause;
    end

    gbid = 7;
    pick = candidates3d(matchgt==gbid&oscf>0.28);
    clf
    vis_point_cloud(points3d,rgb,10,10000);
    hold on;
    vis_cube(gt_bb(gbid),'g');
    for bid = 1:length(pick)
         vis_cube(pick(bid),'b');
        pause;
    end
end


%}

