function dss_marvin_script(gpuid,featype,NYUonly,architecturefile,box_reg,poName,fullbox,preTrainModel,is_combineimg,is_combinehha)
% cd /n/fs/modelnet/deepDetect/Release/code/matlab_code
% dss_marvin_script(0,100,1  ,[]  ,1,'RPN_NYU',1,[],0,0);
% dss_marvin_script(2,100,0,0,'cd',1,'RPN_NYU',1,[],1,0);
opt = dss_initPath();
load('cls.mat');


%%
opt.includegt   = false;
opt.orein_cls   = false;
opt.axisAlign   = false;
opt.is_render   = false;
opt.axisAlign   = false;
opt.num_percate = 0;


context_pad   = 3;
feature_scale = 100;
opt.pos_ratio = 0.25;

opt.tsdf_size      = 30;
opt.lr_stepsize    = 5000;
opt.box_lossweight = 120;
opt.BoxperImage    = 2000;
opt.testminbatch   = 500;
opt.batch_scale    = 3;
%%
opt.encode_type = featype;
opt.GPUhost     = gpuid;
opt.GPUdata     = gpuid;
opt.box_reg     = box_reg;



opt.bb_param_weight = [1,1,1,1,1,1];

opt.fullbox = fullbox;
opt.preTrainModel = preTrainModel;
opt.max_iter = 10000;
opt.iter = opt.max_iter;

min_batchsize = 128*opt.batch_scale;
opt.batch_size = [min_batchsize - round(min_batchsize*opt.pos_ratio), round(min_batchsize*opt.pos_ratio)];
opt.extracfea_batch_size = 512;



opt.scale = feature_scale;
opt.poName = poName;
opt.proposal_dir = [opt.proposal_root '/' opt.poName '/'];
opt.context_pad = context_pad;

opt.numofcate = length(cls)+1;
opt.is_combineimg   = is_combineimg;
opt.is_combinehha   = is_combinehha;

opt.imgfea_dim = 4096;
opt.img_fea_folder = [fullfile(opt.imgfea_dir,opt.poName,'po') '/'];


%% 
opt.feaName = ['dss_' num2str(opt.encode_type)];
if ~isempty(opt.preTrainModel)
   ind = find(opt.preTrainModel== '/');
   preTrainModelName = opt.preTrainModel(ind(end)+1:end);
   opt.feaName =[opt.feaName '_' preTrainModelName];
end

if opt.box_reg
   opt.box_reg = true;
   opt.feaName =[opt.feaName '_breg' num2str(opt.box_lossweight)];
   if isempty(architecturefile)
      architecturefile = 'boxreg';
   end
else
   opt.box_reg = false;
end


if ~strcmp(opt.poName,'rgbd_tight_2')
   opt.feaName =[opt.feaName '_' opt.poName]; 
end

if featype==102 % feature type : [r g b dx dy dz]
    opt.grid_size = [6,opt.tsdf_size,opt.tsdf_size,opt.tsdf_size];
elseif featype==103 % feature type : [tsdf ratio]
    opt.grid_size = [1,opt.tsdf_size,opt.tsdf_size,opt.tsdf_size];
else % feature type : [dx dy dz]
    opt.grid_size = [3,opt.tsdf_size,opt.tsdf_size,opt.tsdf_size];
end


if opt.max_iter~=10000
   opt.feaName =[opt.feaName '_it' num2str(opt.max_iter)];
end
if opt.tsdf_size~=30
   opt.feaName =[opt.feaName '_tds' num2str(opt.tsdf_size)];
   if opt.tsdf_size>30
       opt.extracfea_batch_size = 128;
   end
end


if opt.lr_stepsize ~= 5000
   opt.feaName =[opt.feaName '_stz' num2str(opt.lr_stepsize)];
end
if opt.batch_scale~=1;
   opt.feaName =[opt.feaName '_bs' num2str(opt.batch_scale)];
end
if opt.pos_ratio~=0.25
   opt.feaName =[opt.feaName '_pr' num2str(opt.pos_ratio)];
end

if opt.scale~=100
   opt.feaName =[opt.feaName '_sc' num2str(opt.scale)];
end

if (opt.context_pad~=3)
   opt.feaName =[opt.feaName '_cp' num2str(opt.context_pad)];
end


if opt.is_render 
   opt.feaName =[opt.feaName '_rd'];
end

if ~NYUonly
    opt.feaperbath = opt.BoxperImage/opt.testminbatch;
    opt.feaName =[opt.feaName '_sun'];
    opt.max_iter = max(opt.max_iter,50000);
    opt.loadfea = 0;
else
    opt.loadfea = 1;
    opt.feaperbath = 0;
end




%% genboxfile
if ~exist('architecturefile','var')||isempty(architecturefile)
    architecturefile = fullfile(opt.marvin_dir,'/DSS/define_3d/DSSopt.json');
else
    opt.feaName =[opt.feaName '_' architecturefile];
    architecturefile = fullfile(opt.marvin_dir,['/DSS/define_3d/DSSopt_' architecturefile '.json']);
end


if opt.is_render
    boxfile_traincnn = dss_prepare_synlist(opt.sysn_dataroot,opt.proposal_dir,opt.proposal_dir,cls,opt.BoxperImage);
    boxfile_testcnn  = boxfile_traincnn; 
else 
    proposal_only = 1;
    [boxfile_traincnn] = dss_preparelist('train',NYUonly,opt.proposal_dir,opt.proposal_dir,proposal_only,cls,opt.box_reg,opt.axisAlign,opt.fullbox,opt.BoxperImage,0,0);
    [boxfile_testcnn]  = dss_preparelist('test' ,NYUonly,opt.proposal_dir,opt.proposal_dir,proposal_only,cls,opt.box_reg,opt.axisAlign,opt.fullbox,opt.BoxperImage,0,0);
end
outfilename_gt   = '';
outpath = opt.proposal_dir;
[outfilename_p]  = dss_preparelist('train_test',NYUonly,outpath,opt.proposal_dir, 1,cls,opt.box_reg,opt.axisAlign,opt.fullbox,opt.BoxperImage,0);

%% extract color feature 
if is_combineimg||is_combinehha
    outpath = opt.proposal_dir;
    [box2d_file_p] = dss_preparelist_2d(NYUonly,outpath,opt.proposal_dir,1,cls,opt.BoxperImage);
    template = './marvin test %s %s fc7,da_fc7,cls_prob_0,cls_prob_1 %s/fc7,%s/da_fc7,%s/cls_prob_0,%s/cls_prob_1 1';
    path2trained_color_model = opt.path2trained_color_model;
    extract_imgfeafile_po    = '../pretrainedModels/color_vggvgg_hha_extract_po.json';
    cmdimg_po = sprintf(template,extract_imgfeafile_po, path2trained_color_model, opt.img_fea_folder,...
                        opt.img_fea_folder,opt.img_fea_folder,opt.img_fea_folder);   
    display(cmdimg_po)
end

%% gen define files
define_dir = fullfile(opt.marvin_dir,'/DSS/define_3d/');
mkdir(fullfile(define_dir,opt.feaName));
featureLocalfolder = [opt.feature_tensor_dir '/' opt.feaName 'iter_' num2str(opt.iter) '/'];
mkdir(fullfile(opt.trainedcn_dir,opt.feaName)); 
mkdir(featureLocalfolder);
mkdir([featureLocalfolder '/gt/']);
mkdir([featureLocalfolder '/po/']);


display(architecturefile)
[path2trainedmodel,definefile,extractfeafile_gt,extractfeafile_po] = gendefinejosn...
        (define_dir,opt.trainedcn_dir,architecturefile,boxfile_traincnn,boxfile_testcnn,outfilename_gt,outfilename_p,opt);

%% extract feature file
featurefolder = [opt.feature_mat_dir '/' opt.feaName 'iter_' num2str(opt.iter) '/'];
mkdir(featurefolder);
mkdir([featurefolder '/gt/']);
mkdir([featurefolder '/po/']);
if opt.box_reg
    if opt.orein_cls
        template = './marvin test %s %s fc5,cls_score,box_pred,oreintation %s/fc5.tensor,%s/fc6.tensor,%s/box_pred.tensor,%s/oreintation.tensor';
        cmd_gt = sprintf(template, extractfeafile_gt, path2trainedmodel, fullfile(featureLocalfolder,'gt'), fullfile(featureLocalfolder,'gt'), fullfile(featureLocalfolder,'gt'),fullfile(featureLocalfolder,'gt'));
        cmd_po = sprintf(template, extractfeafile_po, path2trainedmodel, fullfile(featureLocalfolder,'po'), fullfile(featureLocalfolder,'po'), fullfile(featureLocalfolder,'po'),fullfile(featureLocalfolder,'po'));  
    else
        template = './marvin test %s %s fc5,cls_score,box_pred %s/fc5.tensor,%s/fc6.tensor,%s/box_pred.tensor';
        cmd_gt = sprintf(template, extractfeafile_gt, path2trainedmodel,fullfile(featureLocalfolder,'gt'),fullfile(featureLocalfolder,'gt'),fullfile(featureLocalfolder,'gt'));
        cmd_po = sprintf(template, extractfeafile_po, path2trainedmodel, fullfile(featureLocalfolder,'po'),fullfile(featureLocalfolder,'po'),fullfile(featureLocalfolder,'po'));  
    end
else
    template = './marvin test %s %s fc5,prob %s/fc5.tensor,%s/fc6.tensor';
    cmd_gt = sprintf(template, extractfeafile_gt, path2trainedmodel, fullfile(featureLocalfolder,'gt'),fullfile(featureLocalfolder,'gt'));
    cmd_po = sprintf(template, extractfeafile_po, path2trainedmodel, fullfile(featureLocalfolder,'po'),fullfile(featureLocalfolder,'po'));
end

if opt.feaperbath>0
   cmd_po = sprintf('%s %d',cmd_po,opt.feaperbath);
   cmd_gt = sprintf('%s %d',cmd_gt,opt.feaperbath);
end

%% gen the training script : train.sh
trainscriptfile = fullfile(define_dir,opt.feaName,'train.sh');
%trainscriptstring = sprintf('#!/usr/bin/env sh ;\n');
trainscriptstring = sprintf('cd %s;\nexport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s;',opt.marvin_dir,opt.cudnn_libpath);

if ~exist(path2trainedmodel,'file')
   logfile = [opt.log_dir opt.feaName '.log'];
   if ~isempty(opt.preTrainModel)
      trainscriptstring = sprintf('%s\n ./marvin train %s %s 2>&1 | tee  %s',trainscriptstring,definefile,opt.preTrainModel,logfile);
   else
      trainscriptstring = sprintf('%s\n ./marvin train %s 2>&1 | tee  %s',trainscriptstring,definefile,logfile);
   end
end

if opt.includegt&&~exist(sprintf('%s/fc5.tensor',[featureLocalfolder '/gt/']),'file')
   trainscriptstring = sprintf('%s;\n%s',trainscriptstring,cmd_gt);
end
if ~exist(sprintf('%s/fc5.tensor',[featureLocalfolder '/po/']),'file')
   trainscriptstring = sprintf('%s;\n%s\n',trainscriptstring,cmd_po);
end
fid = fopen(trainscriptfile,'w');
fprintf(fid,trainscriptstring);
fclose(fid);

display(trainscriptstring)



cd(opt.marvin_dir)
system('export PATH=$PATH:/usr/local/cuda/bin');
system('unset LD_LIBRARY_PATH;');
system(sprintf('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s;',opt.cudnn_libpath));
system(trainscriptstring);


fprintf('If system call not work try run following command in terminal directly\n');

trancnnCmd =sprintf('cd %s\n/%s',opt.marvin_dir,trainscriptfile);
display(trancnnCmd);
fprintf('pausing ... please press any bottom to continue ...\n');
cd(opt.basepath);
pause;

%% read feature and train svm
if opt.feaperbath>0
   orgnizefeat_batch(featurefolder,featureLocalfolder,outfilename_p,cls,opt);  
else
   orgnizefeat(featurefolder,featureLocalfolder,outfilename_p,cls,opt);
end

%% run the detection 
runall([opt.feaName 'iter_' num2str(opt.iter)],[],opt.includegt,1,1,opt.box_reg,opt.proposal_dir,NYUonly);
