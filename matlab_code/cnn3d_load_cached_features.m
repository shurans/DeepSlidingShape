function d = cnn3d_load_cached_features(cnn3d_model, seqname,loadgt,loadpo,boxreg)
% load features from : cnn3d_model.conf.feat_cachedir
% return d.class = N*1; d.features = N*feadim 
class =[];
feat_gt =[];
feat_po =[];
boxes =[];
overlap =[];
box_pred =[];
if ~exist('boxreg','var')
    boxreg = 0;
end
if loadgt
    load(fullfile(cnn3d_model.conf.feat_cachedir,'gt',seqname));
    if ~isempty(f.boxes)
        feat_gt = f.feat;
        [~,clsid] = ismember({f.boxes.classname},cnn3d_model.classes);
        class = [class,clsid];
        overlapthis = zeros(length(f.boxes),length(cnn3d_model.classes));
        
        for bi = 1:length(f.boxes)
            if clsid(bi) >0;
                overlapthis(bi,clsid(bi)) = 1;
                f.boxes(bi).gt = 1;
            else
                f.boxes(bi).gt = 0;
            end
        end
        overlap = [overlap;overlapthis]; 
        boxes = [boxes;f.boxes(:)];
        
        toremove = {'labelname','sequenceName','orientation','iou','label','gtBb2D','classname','diff'};
        for k =1:length(toremove)
            try boxes = rmfield(boxes,toremove(k)); end
        end
        if boxreg
           box_pred = [box_pred;f.box_pred];
        end
    end
    % load feature from other folder
    if cnn3d_model.training_opts.combinefea
       for fyid =1:length(cnn3d_model.conf.feat_com)
           load(fullfile(cnn3d_model.conf.feat_root,cnn3d_model.conf.feat_com{fyid},'gt',seqname));
           if ~isempty(f.boxes)
              feat_gt = [feat_gt,f.feat];
           end
       end
    end
end


if loadpo
     load(fullfile(cnn3d_model.conf.feat_cachedir,'po',seqname));
     if ~isempty(f.boxes)
         if isfield(f.boxes,'classname')
            [~,clsid] = ismember({f.boxes.classname},cnn3d_model.classes);
         elseif isfield(f.boxes,'labelname')
             [~,clsid] = ismember({f.boxes.labelname},cnn3d_model.classes);
         else
             clsid =0;
         end
        class = [class,clsid];
        feat_po = f.feat;
        overlapthis = zeros(length(f.boxes),length(cnn3d_model.classes));
        for bi =1:length(clsid)
            if clsid(bi) >0;
                overlapthis(bi,clsid(bi)) = f.boxes(bi).iou;
            end
            f.boxes(bi).gt = 0;
        end
        overlap = [overlap;overlapthis];
        toremove = {'classId','imageNum','ioufull','iou','labelname','conf','classname','diff','box2d_tight','box2d_proj'};
            
    
        for k =1:length(toremove)
            try f.boxes = rmfield(f.boxes,toremove(k));end
        end
        boxes = [boxes;f.boxes];
        if boxreg
           box_pred = [box_pred;f.box_pred];
        end
     end
     % load feature from other folder
     if cnn3d_model.training_opts.combinefea
        for fyid =1:length(cnn3d_model.conf.feat_com)
          load(fullfile(cnn3d_model.conf.feat_root,cnn3d_model.conf.feat_com{fyid},'po',seqname));
          if ~isempty(f.boxes)
              feat_po = [feat_po,f.feat];
           end
        end
     end
end

feat = [feat_gt;feat_po];
if cnn3d_model.training_opts.addboxsize&&~isempty(boxes)
   coeffs=[boxes.coeffs];
   coeffs = reshape(coeffs,3,[])'-0.35;
   feat = [feat,coeffs];
end
d.seqname = seqname;
d.class = class(:);
d.boxes = boxes;
d.box_pred = box_pred;

d.feat = feat;
d.overlap = overlap;
end