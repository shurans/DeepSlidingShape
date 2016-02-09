function res = cnn3d_eval_class(class, boxes, imdb, conf_cache_dir, groundtruthBB, overlap_thresh, nms_thresh,suffix)
% evaluation function for 3d detection
if ~exist('overlap_thresh', 'var') || isempty(overlap_thresh)
  overlap_thresh = 0.25;
end

if ~exist('nms_thresh', 'var') || isempty(nms_thresh)
  nms_thresh = 0.25;
end

savefile =[conf_cache_dir '/' class '_pr_' imdb.name suffix '.mat'];
if ~exist(savefile,'file')
    tps = [];
    fps = [];
    all_boxes   = [];
    all_imageid = [];
    npos = 0;
    % gt and boxes match
    for i = 1:length(imdb.image_ids)
        % get the ground truth
        gt_boxes = get_gt_boxes(groundtruthBB,imdb.image_ids(i), class);
        npos = npos + numel(gt_boxes);

        % nms
        bbox = boxes{i};
        bbox = bbox(:);
        keep = nms3d(bbox, nms_thresh);
        bbox = bbox(keep,:);
        if isempty(bbox)
            continue;
        end
        [~, sort_idx] = sort([bbox.conf], 'descend');
        bbox = bbox(sort_idx);
        if ~isfield(bbox,'o')
            for jj = 1:length(bbox)
                bbox(jj).o =[];
            end
        end
        all_boxes = cat(1, all_boxes, bbox);
        all_imageid =  cat(1, all_imageid, i*ones(length(bbox),1));
        % overlaps
        overlaps = bb3dOverlapCloseForm(bbox, gt_boxes');

        gt_detected = zeros(numel(gt_boxes), 1);
        tp = zeros(numel(bbox), 1);
        fp = zeros(numel(bbox), 1);
        for b = 1 : numel(bbox)
            if isempty(gt_boxes)
                fp(b) = 1;
                continue;
            end
            [maxiou, gt_idx] = max(overlaps(b,:));
            if maxiou >= overlap_thresh
                if ~gt_detected(gt_idx)
                    tp(b) = 1;
                    gt_detected(gt_idx) = 1;
                else
                    fp(b) = 1;
                end
            else
                fp(b) = 1;
            end
        end
        tps = cat(1, tps, tp);
        fps = cat(1, fps, fp);
    end

    assert(numel(tps) == numel(fps));
    assert(numel(tps) == numel(all_boxes));
    % sort again
    [~, sort_idx] = sort([all_boxes.conf], 'descend');
    all_boxes = all_boxes(sort_idx);
    all_imageid = all_imageid(sort_idx);
    tps = tps(sort_idx);
    tp = tps;
    fps = fps(sort_idx);

    fps=cumsum(fps);
    tps=cumsum(tps);
    recall=tps/npos;
    prec=tps./(fps+tps);

    % compute average precision
    ap_auc = xVOCap(recall, prec);

    % save and plot
    %{
    clf
    plot(recall, prec);

    % force plot limits
    ylim([0 1]);
    xlim([0 1]);

    print(gcf, '-djpeg', '-r0', ...
      [conf.cache_dir '/' class '_pr_' imdb.name '.jpg']);
    %}
    res.recall = recall;
    res.prec = prec;
    res.ap_auc = ap_auc;
    fprintf('%s:%f\n',class,res.ap_auc)
    save(savefile, ...
        'res', 'recall', 'prec', 'ap_auc','all_boxes','tp','all_imageid');
else
    load(savefile)
end

end



function ap = xVOCap(rec,prec)
% From the PASCAL VOC 2011 devkit

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
end

%{
function gt_boxes = get_gt_boxes(img, class)

if isempty(img.boxes)
    gt_boxes = [];
    return;
end
gt_classes = {img.boxes.classname};
if strcmp(class, 'book_shelf')
    gt_boxes = img.boxes(strcmp(gt_classes, class) | strcmp(gt_classes, 'bookshelf'));
else 
    gt_boxes = img.boxes(strcmp(gt_classes, class));
end
    %}