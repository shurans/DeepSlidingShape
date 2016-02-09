function boxfile = genboxesfile_trainfea(train_or_test,NYUonly,outpath,proposal_dir,...
                                         proposal_only,cls,rescaleBox)
dataroot = '/n/fs/sun3d/data/';
SUNRGBDtoolboxdir = '/n/fs/modelnet/SUN3DV2/prepareGT/';



if NYUonly
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat')
    alltrain = trainSeq;
    alltest = testSeq;
    filename = 'boxes_NYU_trainfea';
else
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/allsplit.mat')
    filename = 'boxes_SUNrgbd_trainfea';
end
if proposal_only
   filename =[filename '_po'];
end
if strcmp(train_or_test,'train') 
   filename = [filename '_train'];
   allpath = alltrain;
elseif strcmp(train_or_test,'test') 
    allpath = alltest;
    filename = [filename '_test'];
else
    error('hey error');
end

if rescaleBox
   filename = [filename '_rsbox' num2str(rescaleBox)];
end

if length(cls)~=19
   filename = [filename 'cate' num2str(length(cls))];
end
seqnames = cell(1,length(allpath));
for i =1:length(allpath)
    seqnames{i} = getSequenceName(allpath{i});
end

%% start 
if ~exist([outpath '/' filename '.txt'],'file')
    fprintf('out: %s\n',[outpath '/' filename '.txt']);
    load([SUNRGBDtoolboxdir '/Metadata/' 'SUNRGBDMeta_tight.mat']);
    fid = fopen([outpath '/' filename '.txt'],'w');
    totalnumofbox =0;
    for imageNum = 1:length(seqnames)
        fprintf('%d image: %s \n', imageNum,seqnames{imageNum})
        [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
        data = SUNRGBDMeta(ind);
        %% print to text
        fprintf(fid,'# %d\n',imageNum-1);
        fprintf(fid,'%s\n',data.depthpath(length(dataroot)+1:end));
        fprintf(fid,'%s\n',data.rgbpath(length(dataroot)+1:end));
        imsize = size(imread(data.rgbpath));
        fprintf(fid,'%d\n%d\n',imsize(1),imsize(2));
        % print R
        R = data.Rtilt';
        for i =1:8
            fprintf(fid,'%f ',R(i));
        end
        fprintf(fid,'%f\n',R(9));

        % print K
        K = data.K';
        for i =1:8
            fprintf(fid,'%f ',K(i));
        end
        fprintf(fid,'%f\n',K(9));

        % print box
        clear boxes;
        numbox = 0;
        if ~proposal_only
            for bi =1:length(data.groundtruth3DBB_tight)
                if ~isempty(data.groundtruth3DBB_tight(bi))&&ismember(data.groundtruth3DBB_tight(bi).classname,cls)
                    numbox = numbox+1;
                    data.groundtruth3DBB_tight(bi).iou = 1;
                    boxes(numbox) = data.groundtruth3DBB_tight(bi);

                end
            end
            if numbox>0
                boxes = rmfield(boxes,{'sequenceName','orientation','gtBb2D','labelname','label'});
            else
                 boxes =[];
            end
        else
            boxes =[];
        end
        
        ld = load([fullfile(proposal_dir,data.sequenceName) '.mat']);
        boxes_po = ld.candidates3d;
        numbox = numbox+length(boxes_po);
        try boxes_po = rmfield(boxes_po,{'ioufull'});end
        try boxes_po = rmfield(boxes_po,{'conf'});end

        try boxes_po = rmfield(boxes_po,{'imageNum','classId'});end
        try boxes_po.classname =  boxes_po.labelname;end
        % cancatenate 
        try boxes = [boxes(:);boxes_po(:)];end
        numbox = length(boxes);
        fprintf(fid,'%d\n',numbox);

        for bi =1:numbox
            if isfield(boxes(bi),'classname')
                [~,classid]= ismember(boxes(bi).classname,cls);
            else
                [~,classid]= ismember(boxes(bi).labelname,cls);
            end
            overlap =boxes(bi).iou;
            if classid==0
               overlap =0;
            end

            fprintf(fid,'%d ',classid);
            fprintf(fid,'%f ',overlap);
            B = boxes(bi).basis';
            for i =1:9
                fprintf(fid,'%f ',B(i));
            end
            for i =1:3
                fprintf(fid,'%f ',boxes(bi).centroid(i));
            end
            for i =1:3
                fprintf(fid,'%f ',boxes(bi).coeffs(i));
            end
            fprintf(fid,'\n');
            totalnumofbox = totalnumofbox+1;
        end
    end
    fclose (fid);
    boxfile = [outpath '/' filename '.txt'];
    save([outpath '/' filename '.mat'],'totalnumofbox','NYUonly');
else
    boxfile = [outpath '/' filename '.txt'];
    fprintf('return: %s\n',[outpath '/' filename '.txt']);
end
fprintf('out: %s\n',[outpath '/' filename '.txt']);
fprintf('out: %s\n',[outpath '/' filename '.mat']);
end
