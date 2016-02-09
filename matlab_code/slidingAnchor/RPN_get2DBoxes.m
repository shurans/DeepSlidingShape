NYUonly = 1;
load('/n/fs/modelnet/SUN3DV2/prepareGT/Metadata/SUNRGBDMeta_tight_Yaw.mat');

proposal_dir = '/n/fs/modelnet/deepDetect/proposal/RPN_multi_dpcv10.35top150_5000/';
proposal_dir_tosave = '/n/fs/modelnet/deepDetect/proposal/RPN_multi_dpcv10.35top150_5000/';

if NYUonly
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat')
    alltrain = trainSeq;
    alltest = testSeq;
else
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/allsplit.mat')
end

allPath = [alltest,alltrain];
seqnames = cell(1,length(allPath));
for i =1:length(allPath)
    seqnames{i} = getSequenceName(allPath{i});
end

for imageNum = 1:length(seqnames)
     [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
     data = SUNRGBDMeta(ind);
     tosavepath = fullfile(proposal_dir_tosave,data.sequenceName);
     fprintf('%d\n',imageNum)
  
     load([fullfile(proposal_dir,data.sequenceName) '.mat']);
     if ~isfield(candidates3d,'box2d_tight')
         % for each candidates3d get box_proj and box_tight
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
         oscfM_r(removeid,:) =[];
         %{
         for j =1:length(candidates3d)
             candidates3d(j).box2d_tight = crop2DBB(candidates3d(j).box2d_tight,imsize(1),imsize(2));
         end
         %}
         
         %{
             box_tight =reshape([candidates3d.box2d_tight],4,[])';
             ov_tight = bb2dOverlap(box_tight,data.groundtruth3DBB_tight');
             [oscf_3d,matchgt] = max(oscfM_r,[],2);
             [oscf_2d,matchgt] = max(ov_tight,[],2);

             pick   = find(oscf_3d>0.25&oscf_2d<0.5);
             pick_2 = find(oscf_3d<0.25&oscf_2d>0.5);
             for kk =1:length(pick)
                 figure(1);clf;

                 vis_point_cloud(points3d,rgb)
                 hold on;

                 vis_cube(candidates3d(pick(kk)),'r')
                 vis_cube(data.groundtruth3DBB(matchgt(pick(kk))),'g')


                 figure(2);clf;
                 imshow(data.rgbpath);
                 hold on; 
                 rectangle('Position', [candidates3d(pick(kk)).box2d_tight(1) candidates3d(pick(kk)).box2d_tight(2)...
                                        candidates3d(pick(kk)).box2d_tight(3) candidates3d(pick(kk)).box2d_tight(4)],'edgecolor','r');
                 rectangle('Position', [data.groundtruth2DBB_tight(matchgt(pick(kk))).gtBb2D(1) data.groundtruth2DBB_tight(matchgt(pick(kk))).gtBb2D(2)...
                                        data.groundtruth2DBB_tight(matchgt(pick(kk))).gtBb2D(3) data.groundtruth2DBB_tight(matchgt(pick(kk))).gtBb2D(4)],'edgecolor','g');

                 fprintf('%f,%f\n',full(oscf_3d(pick(kk))),oscf_2d(pick(kk))); 
                 pause;

             end

            figure,
            imshow(data.rgbpath);
            hold on; 
            for kk =1:length(data.groundtruth2DBB_tight_proj)
                rectangle('Position', [data.groundtruth2DBB_tight(kk).gtBb2D(1) data.groundtruth2DBB_tight(kk).gtBb2D(2)...
                                       data.groundtruth2DBB_tight(kk).gtBb2D(3) data.groundtruth2DBB_tight(kk).gtBb2D(4)],'edgecolor','r');
            end
         %}  
         ind = find(tosavepath == '/');mkdir(tosavepath(1:ind(end)));      
         save([tosavepath '.mat'],'candidates3d','oscfM_r','gtnames');
     end
 end