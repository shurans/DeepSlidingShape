function cnn3d_model = cnn3d_train(cnn3d_model,imdb,replace)
% TO DO: Get the average norm of the features, scale feature before training
cnn3d_modelszvepath = [cnn3d_model.conf.cache_dir 'cnn3d_model.mat'];
if exist(cnn3d_modelszvepath,'file')&&~replace
   load(cnn3d_modelszvepath);
else
    opts = cnn3d_model.training_opts;

    % Record a log of the training and test procedure
    timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
    mkdir(cnn3d_model.conf.cache_dir)
    diary_file = fullfile(cnn3d_model.conf.cache_dir,[ 'cnn3d_train_' timestamp '.txt']);
    diary(diary_file);
    fprintf('Logging output in %s\n', diary_file);

    fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
    fprintf('Training options:\n');
    disp(opts);
    fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

    % ------------------------------------------------------------------------
    % Create a new model
    % cnn3d_model = cnn3d_loadmodel(cnn3d_model);
    
    %% ------------------------------------------------------------------------
    % Get all positive examples
    mkdir(sprintf('%s/%s/',cnn3d_model.conf.cache_dir, imdb.name));
    save_file = sprintf('%s/%s/gt_pos_layer_5_cache.mat', ...
                        cnn3d_model.conf.cache_dir, imdb.name);
    try
      load(save_file);
      fprintf('Loaded saved positives from ground truth boxes\n');
    catch
      [X_pos, keys_pos] = get_positive_features(imdb, cnn3d_model);
      save(save_file, 'X_pos', 'keys_pos', '-v7.3');
    end
    
    % Init training caches
    caches = cell(1,length(imdb.class_ids));
    for i = imdb.class_ids
      fprintf('%14s has %6d positive instances\n', ...
               imdb.classes{i}, size(X_pos{i},1));
      X_pos{i} = cnn3d_pool5_to_fcX(X_pos{i}, opts.layer, cnn3d_model);
      X_pos{i} = cnn3d_scale_features(X_pos{i}, opts);
      caches{i} = init_cache(X_pos{i}, keys_pos{i},opts);
    end
    % ------------------------------------------------------------------------
    % Train with hard negative mining
    first_time = true;
    for hard_epoch = 1:opts.max_hard_epochs
      for i = 1:length(imdb.image_ids)
        fprintf('hard neg epoch: %d/%d image: %d/%d\n', ...
                 hard_epoch, opts.max_hard_epochs, i, length(imdb.image_ids));

        % Get hard negatives for all classes at once (avoids loading feature cache
        % more than once)
        [X, keys] = sample_negative_features(first_time, cnn3d_model, caches, ...
            imdb, i);

        % Add sampled negatives to each classes training cache, removing
        % duplicates
        for j = imdb.class_ids
          if ~isempty(keys{j})
            if ~isempty(caches{j}.keys_neg)
              [~, ~, dups] = intersect(caches{j}.keys_neg, keys{j}, 'rows');
              assert(isempty(dups));
            end
            caches{j}.X_neg = cat(1, caches{j}.X_neg, X{j});
            caches{j}.keys_neg = cat(1, caches{j}.keys_neg, keys{j});
            caches{j}.num_added = caches{j}.num_added + size(keys{j},1);
          end

          % Update model if
          %  - first time seeing negatives
          %  - more than retrain_limit negatives have been added
          %  - its the final image of the final epoch
          update_iter = mod(i,20)==0;
          is_last_time = (hard_epoch == opts.max_hard_epochs && i == length(imdb.image_ids));
          hit_retrain_limit = (caches{j}.num_added > caches{j}.retrain_limit);
          if (first_time || (hit_retrain_limit&&update_iter) || is_last_time) && ...
              ~isempty(caches{j}.X_neg)
            fprintf('>>> Updating %s detector <<<\n', imdb.classes{j});
            fprintf('Cache holds %d pos examples %d neg examples\n', ...
                    size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
            [new_w, new_b] = update_model(caches{j}, opts);
            cnn3d_model.detectors.W(:, j) = new_w;
            cnn3d_model.detectors.B(j) = new_b;
            caches{j}.num_added = 0;

            z_pos = caches{j}.X_pos * new_w + new_b;
            z_neg = caches{j}.X_neg * new_w + new_b;

            caches{j}.pos_loss(end+1) = opts.svm_C * opts.pos_loss_weight * ...
                                        sum(max(0, 1 - z_pos));
            caches{j}.neg_loss(end+1) = opts.svm_C * sum(max(0, 1 + z_neg));
            caches{j}.reg_loss(end+1) = 0.5 * new_w' * new_w + ...
                                        0.5 * (new_b / opts.bias_mult)^2;
            caches{j}.tot_loss(end+1) = caches{j}.pos_loss(end) + ...
                                        caches{j}.neg_loss(end) + ...
                                        caches{j}.reg_loss(end);

            for t = 1:length(caches{j}.tot_loss)
              fprintf('    %2d: obj val: %.3f = %.3f (pos) + %.3f (neg) + %.3f (reg)\n', ...
                      t, caches{j}.tot_loss(t), caches{j}.pos_loss(t), ...
                      caches{j}.neg_loss(t), caches{j}.reg_loss(t));
            end

            % store negative support vectors for visualizing later
            SVs_neg = find(z_neg > -1 - eps);
            cnn3d_model.SVs.keys_neg{j} = caches{j}.keys_neg(SVs_neg, :);
            cnn3d_model.SVs.scores_neg{j} = z_neg(SVs_neg);

            % evict easy examples
            easy = find(z_neg < caches{j}.evict_thresh);
            caches{j}.X_neg(easy,:) = [];
            caches{j}.keys_neg(easy,:) = [];
            fprintf('  Pruning easy negatives\n');
            fprintf('  Cache holds %d pos examples %d neg examples\n', ...
                    size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
            fprintf('  %d pos support vectors\n', numel(find(z_pos <  1 + eps)));
            fprintf('  %d neg support vectors\n', numel(find(z_neg > -1 - eps)));
          end
        end
        first_time = false;

        if opts.checkpoint > 0 && mod(i, opts.checkpoint) == 0
          save([cnn3d_model.conf.cache_dir 'cnn3d_model'], 'cnn3d_model');
        end
      end
    end

    % save the final model
    save([cnn3d_model.conf.cache_dir 'cnn3d_model'], 'cnn3d_model');
end
end

% ------------------------------------------------------------------------
function [X_pos, keys] = get_positive_features(imdb, cnn3d_model)
% ------------------------------------------------------------------------
X_pos = cell(max(imdb.class_ids), 1);
keys = cell(max(imdb.class_ids), 1);
pos_ovr_thresh = cnn3d_model.training_opts.pos_ovr_thresh;
for i = 1:length(imdb.image_ids)
  tic_toc_print('%s: pos features %d/%d\n', ...
                procid(), i, length(imdb.image_ids));
  if cnn3d_model.training_opts.pos_include_gt
      d = cnn3d_load_cached_features(cnn3d_model,imdb.image_ids{i},1,1);
  else
      d = cnn3d_load_cached_features(cnn3d_model,imdb.image_ids{i},0,1);
  end
  
  for j = imdb.class_ids
    if isempty(X_pos{j})
      X_pos{j} = single([]);
      keys{j} = [];
    end
    if ~isempty(d.class)
        sel = find(d.class == j& d.overlap(:,j) > pos_ovr_thresh);
    else
        sel = [];
    end
    if ~isempty(sel)
      X_pos{j} = cat(1, X_pos{j}, d.feat(sel,:));
      keys{j} = cat(1, keys{j}, [i*ones(length(sel),1) sel]);
    end
  end
end

end

% ------------------------------------------------------------------------
function cache = init_cache(X_pos, keys_pos,opt)
% ------------------------------------------------------------------------
cache.X_pos = X_pos;
cache.X_neg = single([]);
cache.keys_neg = [];
cache.keys_pos = keys_pos;
cache.num_added = 0;
cache.retrain_limit = opt.retrain_limit;
cache.evict_thresh = opt.evict_thresh;
cache.hard_thresh = opt.hard_thresh;

cache.pos_loss = [];
cache.neg_loss = [];
cache.reg_loss = [];
cache.tot_loss = [];
end

function [X_neg, keys] = sample_negative_features(first_time, cnn3d_model, ...
                                                  caches, imdb, ind)
% ------------------------------------------------------------------------
opts = cnn3d_model.training_opts;
d = cnn3d_load_cached_features(cnn3d_model,imdb.image_ids{ind},0,1); % to change back 1,1

class_ids = imdb.class_ids;

if isempty(d.feat)
  X_neg = cell(max(class_ids), 1);
  keys = cell(max(class_ids), 1);
  return;
end

d.feat = cnn3d_pool5_to_fcX(d.feat, opts.layer, cnn3d_model);
d.feat = cnn3d_scale_features(d.feat, opts);

neg_ovr_thresh = opts.neg_ovr_thresh;

if first_time
  for cls_id = class_ids
    I = find(d.overlap(:, cls_id) < neg_ovr_thresh);
    X_neg{cls_id} = d.feat(I,:);
    keys{cls_id} = [ind*ones(length(I),1) I];
  end
else
  zs = bsxfun(@plus, d.feat*cnn3d_model.detectors.W, cnn3d_model.detectors.B);
  for cls_id = class_ids
    z = zs(:, cls_id);
    I = find((z > caches{cls_id}.hard_thresh) & ...
             (d.overlap(:, cls_id) < neg_ovr_thresh));

    % Avoid adding duplicate features
    keys_ = [ind*ones(length(I),1) I];
    if ~isempty(caches{cls_id}.keys_neg) && ~isempty(keys_)
      [~, ~, dups] = intersect(caches{cls_id}.keys_neg, keys_, 'rows');
      keep = setdiff(1:size(keys_,1), dups);
      I = I(keep);
    end

    % Unique hard negatives
    X_neg{cls_id} = d.feat(I,:);
    keys{cls_id} = [ind*ones(length(I),1) I];
  end
end
end


% ------------------------------------------------------------------------
function [w, b] = update_model(cache, opts, pos_inds, neg_inds)
% ------------------------------------------------------------------------
solver = 'liblinear';
liblinear_type = 3;  % l2 regularized l1 hinge loss
%liblinear_type = 5; % l1 regularized l2 hinge loss

if ~exist('pos_inds', 'var') || isempty(pos_inds)
  num_pos = size(cache.X_pos, 1);
  pos_inds = 1:num_pos;
else
  num_pos = length(pos_inds);
  fprintf('[subset mode] using %d out of %d total positives\n', ...
      num_pos, size(cache.X_pos,1));
end
if ~exist('neg_inds', 'var') || isempty(neg_inds)
  num_neg = size(cache.X_neg, 1);
  neg_inds = 1:num_neg;
else
  num_neg = length(neg_inds);
  fprintf('[subset mode] using %d out of %d total negatives\n', ...
      num_neg, size(cache.X_neg,1));
end

switch solver
  case 'liblinear'
    ll_opts = sprintf('-w1 %.5f -c %.5f -s %d -B %.5f', ...
                      opts.pos_loss_weight, opts.svm_C, ...
                      liblinear_type, opts.bias_mult);
    fprintf('liblinear opts: %s\n', ll_opts);
    X = sparse(size(cache.X_pos,2), num_pos+num_neg);
    X(:,1:num_pos) = cache.X_pos(pos_inds,:)';
    X(:,num_pos+1:end) = cache.X_neg(neg_inds,:)';
    y = cat(1, ones(num_pos,1), -ones(num_neg,1));
    llm = liblinear_train(y, X, ll_opts, 'col');
    w = single(llm.w(1:end-1)');
    b = single(llm.w(end)*opts.bias_mult);

  otherwise
    error('unknown solver: %s', solver);
end
end
