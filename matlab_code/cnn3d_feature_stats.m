function [mean_norm, stdd,mean_f] = cnn3d_feature_stats(imdb, cnn3d_model)

save_file = sprintf('%s/feature_stats_%s.mat', ...
                    cnn3d_model.conf.cache_dir, cnn3d_model.cache_name);

try
  ld = load(save_file);
  mean_norm = ld.mean_norm;
  stdd = ld.stdd;
  mean_f = ld.mean_f;
  clear ld;
catch
  % fix the random seed for repeatability
  prev_rng = seed_rand();

  image_ids = imdb.image_ids;

  num_images = min(length(image_ids), 200);
  boxes_per_image = 200;

  image_ids = image_ids(randperm(length(image_ids), num_images));

  ns = [];
  sumF =0;
  count =0;
  for i = 1:length(image_ids)
    tic_toc_print('feature stats: %d/%d\n', i, length(image_ids));

    %d = rcnn_load_cached_pool5_features(rcnn_model.cache_name, imdb.name, image_ids{i});
    d = cnn3d_load_cached_features(cnn3d_model, image_ids{i},cnn3d_model.training_opts.pos_include_gt,1);
    X = d.feat(randperm(size(d.feat,1), min(boxes_per_image, size(d.feat,1))), :);
    X = cnn3d_pool5_to_fcX(X, cnn3d_model.training_opts.layer, cnn3d_model);
    sumF = sumF+sum(X,1);
    count = count +size(X,1);
    ns = cat(1, ns, sqrt(sum(X.^2, 2)));
  end

  mean_norm = mean(ns);
  mean_f = sumF/count;
  stdd = std(ns);
  save(save_file, 'mean_norm', 'stdd','mean_f');
  % restore previous rng
  rng(prev_rng);
end