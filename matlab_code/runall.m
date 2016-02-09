function runall(feaName,cls,pos_include_gt,rmbadsize,trainsvm,box_reg,proposal_dir,NYUonly)
% Example:
% runall('dss100_breg120_RPN_multi_dpcv10.35top150_5000_fix_bs3_boxregiter_10000',[],0,1,1,1,'/n/fs/modelnet/deepDetect/proposal/RPN_multi_dpcv10.35top150_5000_fix/',1)
dbstop if error
if ~exist('trainsvm','var')
    trainsvm = 0;
end
load('cls.mat');
amodalbox = 1; % use amodal box or modal box 
replace =0;
[cnn3d_model] = dss_config(feaName,pos_include_gt,rmbadsize,box_reg,proposal_dir,amodalbox);
mkdir(cnn3d_model.conf.cache_dir);
cnn3d_model.classes = cls;



imdbtrain = dss_cread_imdb(NYUonly,'train',cls);
cnn3d_model.imdbname = imdbtrain.name;
if NYUonly
    imdbtest = dss_cread_imdb(NYUonly,'test',cls);
else
    trainsvm =0;
    imdbtest = dss_cread_imdb(NYUonly,'test',cls);
end
imdbeval = dss_cread_imdb(NYUonly,'test',cls);

suffix = '';
if box_reg 
   suffix =[suffix '_breg'];
end
if rmbadsize 
   suffix =[suffix '_rmbadsize'];
end
if trainsvm
    % Get the average norm of the features
    [cnn3d_model.training_opts.feat_norm_mean] = cnn3d_feature_stats(imdbtrain, cnn3d_model);
    fprintf('average norm = %.3f\n', cnn3d_model.training_opts.feat_norm_mean);
    cnn3d_model = cnn3d_train(cnn3d_model,imdbtrain,replace);
else
    suffix = [suffix '_nosvm'];
end

% test
if ~box_reg
    cnn3d_model.training_opts.box_reg =0;
else
    suffix = [suffix '_breg'];
    cnn3d_model.training_opts.box_reg =1;
    cnn3d_model.training_opts.proposal_dir = proposal_dir;
end
if rmbadsize 
   cnn3d_model.training_opts.rb_dis = 1;
else
   cnn3d_model.training_opts.rb_dis = 0;    
end
cnn3d_test_size(cnn3d_model,imdbtest,trainsvm,suffix);

% eval
overlap_thresh = 0.25;
nms_thresh     = 0.1;
load(fullfile(cnn3d_model.conf.SUNrgbd_toolbox,'Metadata/SUNRGBDMeta.mat'));
if amodalbox
    groundtruthBB = [SUNRGBDMeta.groundtruth3DBB];
else
    groundtruthBB = [SUNRGBDMeta.groundtruth3DBB_tight];
end

for i = 1:length(cnn3d_model.classes)
    load([cnn3d_model.conf.cache_dir cnn3d_model.classes{i} '_boxes_' imdbtest.name suffix]);
    [~,order]=ismember(imdbeval.image_ids,imdbtest.image_ids);
    boxes = boxes(order);
    cnn3d_eval_class(cnn3d_model.classes{i} , boxes, imdbeval, cnn3d_model.conf.cache_dir,groundtruthBB,overlap_thresh, nms_thresh,suffix);
end

ap = 0;
for i =1:length(cnn3d_model.classes)
    load([cnn3d_model.conf.cache_dir '/' cnn3d_model.classes{i} '_pr_' imdbtest.name suffix])
    fprintf('%f\n',res.ap_auc);
    ap = ap+res.ap_auc;
end
fprintf('mAP :%f\n',ap/length(cnn3d_model.classes))

visulization(SUNRGBDMeta,cnn3d_model,...
             cnn3d_model.conf.cache_dir(length('./cache/')+1:end),imdbtest,suffix);

