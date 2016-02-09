function [path2trainedmodel,definefile,extractfeafile_gt,extractfeafile_po] = ...
    gendefinejosn(define_dir,trainedcn_dir,architecturefile,boxfile_traincnn,boxfile_testcnn,outfilename_gt,outfilename_p,opt)

snapshot_prefix = fullfile(trainedcn_dir,opt.feaName);
if opt.iter<opt.max_iter
    path2trainedmodel = [snapshot_prefix '/DSSnet_snapshot_' num2str(opt.iter) '.marvin'];
else
    path2trainedmodel = [snapshot_prefix '/DSSnet.marvin'];
end
jsonopt.ParseLogical = 1;
%%
defaultopt = loadjson(architecturefile);
%defaultopt.path = [snapshot_prefix '/DSSnet'];
defaultopt.train.snapshot_prefix = [snapshot_prefix '/snapshot'];
defaultopt.train.max_iter = opt.max_iter;
defaultopt.train.GPU = opt.GPUhost;
defaultopt.train.lr_stepsize = opt.lr_stepsize;
defaultopt.train.path =[snapshot_prefix '/DSSnet'];

defaultopt.train.GPU = opt.GPUhost;
defaultopt.test.GPU  = opt.GPUhost;
fc_cls_layid =0;
fc_box_pred_layid =0;
loss_bbox_layid = 0;
for layerid =1:length(defaultopt.layers)
    if strcmp(defaultopt.layers{layerid}.type,'Softmax')
        if (defaultopt.layers{layerid}.stable_gradient)
           defaultopt.layers{layerid}.stable_gradient = true;
        end
    end
    if strcmp(defaultopt.layers{layerid}.name,'fc_cls')|| strcmp(defaultopt.layers{layerid}.name,'fc_cls_20')
        fc_cls_layid = layerid;
    end
    if strcmp(defaultopt.layers{layerid}.name,'fc_box_pred')
        fc_box_pred_layid = layerid;
    end
    if strcmp(defaultopt.layers{layerid}.name,'loss_bbox')
        loss_bbox_layid = layerid;
    end
    
end
defaultopt.layers{fc_cls_layid}.num_output = opt.numofcate;
if fc_box_pred_layid>0
   defaultopt.layers{fc_box_pred_layid}.num_output = opt.numofcate*6; 
end

if loss_bbox_layid>0
    defaultopt.layers{loss_bbox_layid}.loss_weight = opt.box_lossweight; 
end

for layerid = 1:2
    defaultopt.layers{layerid}.batch_size       = opt.batch_size;
    defaultopt.layers{layerid}.encode_type      = opt.encode_type;
    defaultopt.layers{layerid}.context_pad      = opt.context_pad;
    defaultopt.layers{layerid}.scale            = opt.scale;
    
    defaultopt.layers{layerid}.grid_size        = opt.grid_size;
    defaultopt.layers{layerid}.GPU              = opt.GPUdata;
    defaultopt.layers{layerid}.num_categories   = opt.numofcate;
    defaultopt.layers{layerid}.bb_param_weight  = opt.bb_param_weight;
    defaultopt.layers{layerid}.box_reg          = opt.box_reg;
    defaultopt.layers{layerid}.num_percate      = opt.num_percate;
    
    if opt.is_render
       defaultopt.layers{layerid}.is_render     = true;
       defaultopt.layers{layerid}.data_root     = opt.sysn_dataroot;
    else
       defaultopt.layers{layerid}.data_root     = opt.data_root;
       defaultopt.layers{layerid}.is_render     = false;
    end
    
    defaultopt.layers{layerid}.is_combineimg    = boolean(opt.is_combineimg);
    defaultopt.layers{layerid}.is_combinehha    = boolean(opt.is_combinehha);
    defaultopt.layers{layerid}.img_fea_folder   = opt.img_fea_folder;
	defaultopt.layers{layerid}.imgfea_dim       = opt.imgfea_dim;
    defaultopt.layers{layerid}.box_2dreg        = boolean(false);
    defaultopt.layers{layerid}.orein_cls        = boolean(false);
end

defaultopt.layers{1}.file_list = boxfile_traincnn;
defaultopt.layers{2}.file_list = boxfile_testcnn;



for layerid =1:length(defaultopt.layers)
    if strcmp(defaultopt.layers{layerid}.type,'Convolution')||strcmp(defaultopt.layers{layerid}.type,'InnerProduct')
        defaultopt.layers{layerid}.train_me = true;
    end
end

definefile = fullfile(define_dir,opt.feaName,'DSSnet.json');
jsonopt.FileName = definefile;
savejson('',defaultopt,jsonopt);


%% extract fea json gt 
defaultopt.layers(1)                 = [];
defaultopt.layers{1}.batch_size      = opt.testminbatch;
defaultopt.layers{1}.file_list       = outfilename_gt;
defaultopt.layers{1}.data_root       = opt.data_root;
defaultopt.layers{1}.is_render       = false;

extractfeafile_gt = fullfile(define_dir,opt.feaName,'DSS_extractfea_gt.json');
jsonopt.FileName = extractfeafile_gt;
for layerid =1:length(defaultopt.layers)
    if strcmp(defaultopt.layers{layerid}.type,'Convolution')||strcmp(defaultopt.layers{layerid}.type,'InnerProduct')
       defaultopt.layers{layerid}.train_me = false;
    end
end
savejson('',defaultopt,jsonopt);

%% extract fea json po 
defaultopt.layers{1}.file_list = outfilename_p;
extractfeafile_po = fullfile(define_dir,opt.feaName,'DSS_extractfea_po.json');
jsonopt.FileName = extractfeafile_po;
for layerid =1:length(defaultopt.layers)
    if strcmp(defaultopt.layers{layerid}.type,'Convolution')||strcmp(defaultopt.layers{layerid}.type,'InnerProduct')
       defaultopt.layers{layerid}.train_me = false;
    end
end
savejson('',defaultopt,jsonopt);