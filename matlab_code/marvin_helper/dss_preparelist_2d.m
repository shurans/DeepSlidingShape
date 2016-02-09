function [boxfile,totalnumofbox] = dss_preparelist_2d(NYUonly,outpath,proposal_dir,proposal,cls,BoxperImage)
SUNRGBDtoolboxdir = '/n/fs/modelnet/SUN3DV2/prepareGT/';
gt = ~proposal;
if NYUonly
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat')
    alltrain = trainSeq;
    alltest = testSeq;
    filename = 'boxes2d_NYU';
else
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/allsplit.mat')
    filename = 'boxes2d_SUNrgbd';
end

allpath = [alltest,alltrain];
writeTarget = 1;

if proposal
   filename =[filename '_po'];
else
   filename =[filename '_gt'];
end

if BoxperImage>0
   filename = [filename '_nb' num2str(BoxperImage)];
end

if length(cls)~=19
   filename = [filename 'cate' num2str(length(cls))];
end

seqnames = cell(1,length(allpath));
for i =1:length(allpath)
    seqnames{i} = getSequenceName(allpath{i});
end

boxfile = [outpath '/' filename '.list'];
if ~exist(boxfile,'file')
    fprintf('out: %s\n',boxfile);
    load([SUNRGBDtoolboxdir '/Metadata/' 'SUNRGBDMeta_tight_Yaw.mat']);
    fid = fopen(boxfile,'wb');
    filelog =[];
    totalnumofbox =0;
    for imageNum = 1:length(seqnames)
        fprintf('%d image: %s \n', imageNum,seqnames{imageNum})
        [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
        data = SUNRGBDMeta(ind);
        %% get boxes
        clear boxes;
        numbox = 0;
        if gt
            for bi =1:length(data.groundtruth3DBB)
                [~,box2d_tight_id] = ismember(bi,[data.groundtruth2DBB_tight.box3didx]);
                if ~isempty(data.groundtruth3DBB(bi))&&ismember(data.groundtruth3DBB(bi).classname,cls)&&box2d_tight_id>0
                    numbox = numbox+1;
                    
                    boxes(numbox).iou           = 1;
                    boxes(numbox).classname     = data.groundtruth3DBB(bi).classname;
                    boxes(numbox).box2d_tight   = data.groundtruth2DBB_tight(box2d_tight_id).gtBb2D;
                    boxes(numbox).box2d_proj    = data.groundtruth2DBB_full(bi).gtBb2D;
                end
            end
            if numbox==0
               boxes =[];
            end
            candidatefilename = boxes;
        end
        
        if proposal
            candidatefilename = [fullfile(proposal_dir,data.sequenceName) '.mat'];
            ld = load([fullfile(proposal_dir,data.sequenceName) '.mat']);
            boxes = ld.candidates3d;
            if BoxperImage>0
               boxes = boxes(1:min(BoxperImage,length(boxes)));
            end
            filelog(imageNum).boxesfile = candidatefilename;
            filelog(imageNum).seqname = seqnames{imageNum};
        end
        
        
        fwrite(fid, uint32(length(data.sequenceName)), 'uint32');
        fwrite(fid, data.sequenceName, 'char*1');
        R = data.Rtilt';
        fwrite(fid, single(R), 'single'); 
        K = data.K';
        fwrite(fid, single(K), 'single'); 
        imsize = size(imread(data.rgbpath));
        fwrite(fid, uint32(imsize(1)), 'uint32'); 
        fwrite(fid, uint32(imsize(2)), 'uint32');
        
        %% print boxes
        numbox = length(boxes);
        fwrite(fid, uint32(numbox), 'uint32');
        for bi =1:numbox
            [~,classid]= ismember(boxes(bi).classname,cls);
            overlap =boxes(bi).iou;
            if classid==0
               overlap =0;
            end
            if overlap<0.25
               classid =0;
            end
            
            % write 2d tight box
            if isempty(boxes(bi).box2d_tight)
               boxes(bi).box2d_tight = crop2DBB(boxes(bi).box2d_proj,imsize(1),imsize(2));
            else
               boxes(bi).box2d_tight = crop2DBB(boxes(bi).box2d_tight,imsize(1),imsize(2));
            end
            fwrite(fid,uint32(classid), 'uint32');
            box2d_tight_tblr = [boxes(bi).box2d_tight([1,2]),boxes(bi).box2d_tight([1,2])+boxes(bi).box2d_tight([3,4])];
            box2d_tight_tblr = round(box2d_tight_tblr([2,4,1,3]))-1;
            fwrite(fid,single(box2d_tight_tblr), 'single');
            fwrite(fid,uint8(0), 'uint8');
            
            % write 2d full box
            box2d_proj_tblr = [boxes(bi).box2d_proj([1,2]),boxes(bi).box2d_proj([1,2])+boxes(bi).box2d_proj([3,4])];
            box2d_proj_tblr = round(box2d_proj_tblr([2,4,1,3]))-1;
            fwrite(fid,single(box2d_proj_tblr), 'single');
            fwrite(fid,uint8(0), 'uint8');
            
            totalnumofbox = totalnumofbox+1;
        end

    end
    fclose (fid);
    fprintf('saving to mat...\n')
    save([outpath '/' filename '.mat'],'BoxperImage','filelog','totalnumofbox','gt','NYUonly','-v7.3');
else
    load([outpath '/' filename '.mat'],'totalnumofbox');
    fprintf('return: %s\n',boxfile);
end
if 0
    fid = fopen('/n/fs/modelnet/deepDetect/proposal/RPN_multi_dpcv10.35top150_5000/boxes2d_NYU_po_nb2000.list')
    
    len = fread(fid , 1, 'int32');	
    name = fread(fid , len,  'char*1');	
    name = [char(name')];
    display(name)
    R = fread(fid , 9, 'single');	
    K = fread(fid , 9, 'single');	
    h = fread(fid , 1, 'uint32');	
    w = fread(fid , 1, 'uint32');	
    len = fread(fid , 1, 'uint32');

    for i =1:len
         box(i).cls = fread(fid , 1, 'uint32');
         box(i).tblr = fread(fid , 4, 'single');	
         hasgt = fread(fid , 1, 'uint8');
         box(i).tblrf = fread(fid , 4, 'single');	
         hasgt = fread(fid , 1, 'uint8');
    end
end
end

