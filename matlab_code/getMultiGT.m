clear all;
addpath /n/fs/modelnet/slidingShape_release_all/code_benchmark;
setup_benchmark;
sysn_dataroot = '/n/fs/sunhome/SUNRGBDRender_results/';
load('/n/fs/modelnet/SUN3DV2/prepareGT/Metadata/SUNRGBDMeta_tight_Yaw.mat');
load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/allsplit.mat');
allPath = [alltest,alltrain];
check =[];
for ind = 1:length(SUNRGBDMeta)
    data = SUNRGBDMeta(ind);
    
    data_3d = readframeSUNRGBD_folder(fullfile('/n/fs/sun3d/data/',data.sequenceName),[],[],'annotationY');
    data_2d = readframeSUNRGBD_2d3d(fullfile('/n/fs/sun3d/data/',data.sequenceName));
    
    clear groundtruth2DBB_tight_proj groundtruth3DBB  groundtruth2DBB_full groundtruth2DBB_tight
    
    if length(data_3d.groundtruth3DBB)>length(data.groundtruth3DBB)
        groundtruth3DBB = data_3d.groundtruth3DBB;
        [rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
        for j =1:length(groundtruth3DBB)
            [gtBb2D,label,gtBb2Dproject] = get2Dbbwithdepth(groundtruth3DBB(j),imsize,points3d,depthInpaint);
            groundtruth2DBB_tight_proj(j).gtBb2D = gtBb2D;
        end
        
        if sum(abs(data_3d.Rtilt(:) - data.Rtilt(:)))>0.1
            for k =1:length(data_3d.groundtruth3DBB)
                groundtruth3DBB(k).centroid = [data.Rtilt*data_3d.Rtilt'*data_3d.groundtruth3DBB(k).centroid']';
                groundtruth3DBB(k).basis = [data.Rtilt*data_3d.Rtilt'*data_3d.groundtruth3DBB(k).basis']';
            end
           check = [check,ind];
        end 
    else
        groundtruth3DBB = data.groundtruth3DBB;
        groundtruth2DBB_tight_proj = data.groundtruth2DBB;
    end
    % remove the 3d ground truth outside FOV or has no points inside 
    removeid = [];
    for i =1:length(groundtruth3DBB)
        if isempty(groundtruth2DBB_tight_proj(i).gtBb2D)
            removeid = [removeid,i];
        end
    end
    groundtruth3DBB(removeid) = [];
    groundtruth2DBB_tight_proj(removeid) = [];
    
    if ~isempty(groundtruth3DBB)
        % for each groundtruth3DBB find groundtruth2DBB_full box using CG model
        %{
        allsysnfile = dir([fullfile(sysn_dataroot,data.sequenceName,'fixed_angle') '/*.mat']);
        if ~isempty(allsysnfile)
           load(fullfile(sysn_dataroot,data.sequenceName,'fixed_angle',allsysnfile(1).name) );
        else
            obj_metadata =[];
        end
        %}
        for i =1:length(groundtruth3DBB)
            %if ~isempty(obj_metadata)&&~isempty(obj_metadata(i))&&(i<length(obj_metadata))
            %    obj = loadobj(fullfile('/n/fs/sunhome/Yinda/input/',obj_metadata(i).class,obj_metadata(i).modelID))
            %else
            [bb2d,bb2dDraw] = projectStructBbsTo2d(groundtruth3DBB(i),data.Rtilt,[],data.K);
            %bb2d(1:2) = bb2d(1:2)+bb2d(3:4)*0.05;
            %bb2d(3:4) = bb2d(3:4)*0.9;
            groundtruth2DBB_full(i).gtBb2D  = bb2d(1:4);
        end

        % find the match for each 3D boxes
        groundtruth2DBB_tight = data_2d.groundtruth2DBB;
        
        if ~isempty(groundtruth2DBB_tight)
            ov = bb2dOverlap(groundtruth2DBB_tight',groundtruth2DBB_tight_proj');
            classall = unique([{groundtruth2DBB_tight.classname} {groundtruth3DBB.classname}]);
            [~,cid_2DBB_tight] = ismember({groundtruth2DBB_tight.classname},classall);
            [~,cid_3DBB] = ismember({groundtruth3DBB.classname},classall);
            isSameclass = bsxfun(@eq,cid_2DBB_tight',cid_3DBB);
            ov(~isSameclass) = 0;
            [maxoverlap, matchboxid] = max(ov,[],2);
            for boxid =1:length(groundtruth2DBB_tight)
                [has3dbox,gtid] = ismember(boxid,matchboxid);
                if maxoverlap(boxid)>0.1&&has3dbox
                   groundtruth2DBB_tight(boxid).box3didx = gtid;
                else
                   groundtruth2DBB_tight(boxid).box3didx = 0;
                end
            end
            groundtruth2DBB_tight = rmfield(groundtruth2DBB_tight,{'objid','has3dbox'});
        end

        % save result
        groundtruth3DBB = rmfield(groundtruth3DBB,{'gtBb2D','labelname'});
    else
        groundtruth2DBB_tight =[];
        groundtruth2DBB_full =[];
    end
    data = rmfield(data,{'groundtruth2DBB','groundtruth3DBB'});
    
    data.groundtruth2DBB_tight = groundtruth2DBB_tight;
    data.groundtruth2DBB_full  = groundtruth2DBB_full;
    data.groundtruth2DBB_tight_proj = groundtruth2DBB_tight_proj;
     
    data.groundtruth3DBB       = groundtruth3DBB;
    
    
    SUNRGBDMeta_multi(ind) = data;
    % visulize 
    if 0%mod(ind,100)==55
        figure,
        imshow(data.rgbpath);
        hold on; 
        for kk =1:length(data.groundtruth2DBB_tight_proj)
            rectangle('Position', [data.groundtruth2DBB_tight_proj(kk).gtBb2D(1) data.groundtruth2DBB_tight_proj(kk).gtBb2D(2)...
                                   data.groundtruth2DBB_tight_proj(kk).gtBb2D(3) data.groundtruth2DBB_tight_proj(kk).gtBb2D(4)],'edgecolor','r');
        end
        
        hold on; 
        for kk =1:length(data.groundtruth2DBB_tight)
            rectangle('Position', [data.groundtruth2DBB_tight(kk).gtBb2D(1) data.groundtruth2DBB_tight(kk).gtBb2D(2)...
                                   data.groundtruth2DBB_tight(kk).gtBb2D(3) data.groundtruth2DBB_tight(kk).gtBb2D(4)],'edgecolor','y');
        end
        
        for kk =1:length(data.groundtruth2DBB_full)
            rectangle('Position', [data.groundtruth2DBB_full(kk).gtBb2D(1) data.groundtruth2DBB_full(kk).gtBb2D(2) ...
                                   data.groundtruth2DBB_full(kk).gtBb2D(3) data.groundtruth2DBB_full(kk).gtBb2D(4)],'edgecolor','g');
        end

        figure,
        [rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
        vis_point_cloud(points3d,rgb)
        hold on;
        for kk =1:length(data.groundtruth3DBB)
           vis_cube(data.groundtruth3DBB(kk),'r')
        end
    end
    
end
SUNRGBDMeta = SUNRGBDMeta_multi;
save('/n/fs/modelnet/SUN3DV2/prepareGT/Metadata/SUNRGBDMeta_multi.mat','SUNRGBDMeta','check');
