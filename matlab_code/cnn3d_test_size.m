function cnn3d_test_size(cnn3d_model,imdb,trainsvm,suffix)
feat_opts = cnn3d_model.training_opts;
conf = cnn3d_model.conf;
display(feat_opts)
display(conf)
if feat_opts.box_reg
   mean_std = load(fullfile(feat_opts.proposal_dir,'mean_std'),'means','stds');
end
onlygtflag = cnn3d_model.training_opts.onlygtbox;
num_classes = length(cnn3d_model.classes);
image_ids = imdb.image_ids;

max_per_set = ceil(500000/2500)*length(image_ids);
max_per_image = 300;
top_scores = cell(num_classes, 1);
thresh = -inf(num_classes, 1);
box_counts = zeros(num_classes, 1);
vis = 0;
%% skip?
skip =1;
for i =1:length(cnn3d_model.classes)
    if ~exist([conf.cache_dir cnn3d_model.classes{i} '_boxes_' imdb.name suffix '.mat'],'file')
        skip =0;
        break;
    end
end
if skip
   return;
end

if feat_opts.rb_dis
    for j = 1:num_classes
        ld = load([conf.sizemodel_dir '/' cnn3d_model.classes{j} '.mat']);
        sizeModel.allhist(:,:,j) = ld.allhist;
        sizeModel.allbin(:,:,j) = ld.allbin;
    end
end

%% test for each box
count = 0;
for i =1:length(imdb.image_ids)
    fprintf('%d/%d testing on image : %s \n',i,length(imdb.image_ids),imdb.image_ids{i});
    count = count + 1;
    
    if trainsvm
        d = cnn3d_load_cached_features(cnn3d_model,imdb.image_ids{i},0,~onlygtflag,feat_opts.box_reg); % to change
        if isempty(d.feat)
            continue;
        end
        d.feat = cnn3d_pool5_to_fcX(d.feat, feat_opts.layer, cnn3d_model);
        d.feat = cnn3d_scale_features(d.feat, feat_opts);
        zs = bsxfun(@plus, d.feat*cnn3d_model.detectors.W, cnn3d_model.detectors.B);
    else
        load(fullfile(cnn3d_model.conf.feat_cachedir,'po',imdb.image_ids{i}));
        zs = f.score(:,2:end);% first score is back ground score
    end
    
    for j = 1:num_classes
        if feat_opts.box_reg
           boxes = getBox_reg(d,j,mean_std);
        else
           boxes = d.boxes; 
        end
        z = zs(:,j);
        % remove box with too strange size
        if feat_opts.rb_dis
           to_remove = removeBoxsizedis_cls(boxes,sizeModel,j);
           z = z - 2*to_remove;
        end
        
        I = find(z > thresh(j));
        boxes = boxes(I);
        scores = z(I);
        [~, ord] = sort(scores, 'descend');
        ord = ord(1:min(length(ord), max_per_image));
        boxes = boxes(ord);
        scores = scores(ord);
        for bi =1 :length(boxes)
            boxes(bi).conf = scores(bi);
        end
        aboxes{j}{i} = boxes;

        box_counts(j) = box_counts(j) + length(boxes);
        top_scores{j} = cat(1, top_scores{j}, scores);
        top_scores{j} = sort(top_scores{j}, 'descend');
        if box_counts(j) > max_per_set
          top_scores{j}(max_per_set+1:end) = [];
          thresh(j) = top_scores{j}(end);
        end
    end
    %{
    if vis
     cnn3d_model.conf.dataRoot ='/n/fs/sun3d/data/';
     data = readframeSUNRGBD([cnn3d_model.conf.dataRoot '/' imdb.image_ids{i}]);
     [rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
     vis_point_cloud(points3d,rgb,10,10000);
     hold on;
     for j = 1:num_classes
         for bi = 1:length(aboxes{j}{i})
             if aboxes{j}{i}(bi).conf>-0.97
                vis_cube(aboxes{j}{i}(bi),ObjectColor(j),2,cnn3d_model.classes{j});
                pause;
             end
         end
     end
    end
    %}
end

for i = 1:num_classes
    % go back through and prune out detections below the found threshold
    for j = 1:length(image_ids)
      if ~isempty(aboxes{i}{j})
        I = find([aboxes{i}{j}.conf] < thresh(i));
        aboxes{i}{j}(I) = [];
      end
    end

    save_file = [conf.cache_dir cnn3d_model.classes{i} '_boxes_' imdb.name suffix];
    boxes = aboxes{i};
    save(save_file, 'boxes');
    clear boxes;
end
end
