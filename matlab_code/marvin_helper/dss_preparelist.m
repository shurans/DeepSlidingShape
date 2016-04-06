function [boxfile,totalnumofbox] = dss_preparelist(train_or_test,NYUonly,outpath,proposal_dir,...
                                                   proposal,cls,boxreg,axisAlign,fullbox,BoxperImage,box_2dreg,orein_cls)
SUNRGBDtoolboxdir = './external/SUNRGBDtoolbox/';
gt = ~proposal;
writeTarget = 0;
write2dTarget = 0;




if NYUonly
    load(fullfile(SUNRGBDtoolboxdir,'traintestSUNRGBD/test_kv1NYU.mat'))
    load(fullfile(SUNRGBDtoolboxdir,'traintestSUNRGBD/train_kv1NYU.mat'))
    alltrain = trainSeq;
    alltest = testSeq;
    filename = 'boxes_NYU';
else
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/allsplit.mat')
    filename = 'boxes_SUNrgbd';
end

if proposal
   filename =[filename '_po'];
else
   filename =[filename '_gt'];
end

if strcmp(train_or_test,'train') 
   filename = [filename '_train'];
   allpath = alltrain;
   if boxreg
      writeTarget = 1;
      filename = [filename '_diff'];
      mean_std = load(fullfile(proposal_dir,'mean_std'));
      if box_2dreg
         write2dTarget =1;
         filename = [filename '_2d3d'];
      end
      
      if orein_cls
         writeorientation =1;
         filename = [filename '_orein'];
      end
   end
elseif strcmp(train_or_test,'test') 
    allpath = alltest;
    filename = [filename '_test'];
    writeTarget = 0;
else strcmp(train_or_test,'train_test')
    allpath = [alltest,alltrain];
    writeTarget = 0;
end

if BoxperImage>0
   filename = [filename '_nb' num2str(BoxperImage)];
end
if fullbox
    filename = [filename '_fb'];
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
                if ~isempty(data.groundtruth3DBB(bi))&&ismember(data.groundtruth3DBB(bi).classname,cls)
                    numbox = numbox+1;
                    data.groundtruth3DBB(bi).iou = 1;
                    boxes(numbox) = data.groundtruth3DBB(bi);
                end
            end
            if numbox>0
                boxes = rmfield(boxes,{'sequenceName','orientation','gtBb2D','labelname','label'});
            else
                boxes =[];
            end
            candidatefilename = boxes;
        end
        
        if proposal
            candidatefilename = [fullfile(proposal_dir,data.sequenceName) '.mat'];
            ld = load([fullfile(proposal_dir,data.sequenceName) '.mat']);
            boxes = ld.candidates3d;
            if BoxperImage>0
                if BoxperImage<=length(boxes)
                   boxes = boxes(1:min(BoxperImage,length(boxes)));
                else
                   fprintf('%s padding %d to %d \n',length(boxes),BoxperImage);
                   for padi = 1:BoxperImage-length(boxes)
                       boxes = [boxes;boxes(1)];
                   end
                end
            end
            try boxes.classname =  boxes.labelname;end
        end
       
        filelog(imageNum).boxesfile = candidatefilename;
        filelog(imageNum).seqname = seqnames{imageNum};
        
        if axisAlign
           boxes =  makeaxisAlign(boxes);
        end
        %% print to text
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


            fwrite(fid,uint32(classid), 'uint32');
            B = boxes(bi).basis';
            fwrite(fid,single(B), 'single');
            fwrite(fid,single(boxes(bi).centroid),'single');
            fwrite(fid,single(boxes(bi).coeffs),'single');
            totalnumofbox = totalnumofbox+1;
            
            if writeTarget
               if  classid>0
                   fwrite(fid,uint8(1), 'uint8'); % uint8
                   diff = boxes(bi).diff;
                   diff = diff - mean_std.means(classid,:);
                   diff = diff./mean_std.stds(classid,:);
                   fwrite(fid,single(diff(:)), 'single');
                   
                   
                   if write2dTarget
                      if ~isempty(boxes(bi).diff_2dt)
                         diff_2dt = (boxes(bi).diff_2dt- mean_std.means_2d_t(classid,:))./mean_std.stds_2d_t(classid,:);
                      else
                         diff_2dt = [100,100,100,100];
                      end
                      if ~isempty(boxes(bi).diff_2dp)
                         diff_2dp = (boxes(bi).diff_2dp- mean_std.means_2d_p(classid,:))./mean_std.stds_2d_p(classid,:);
                      else
                         diff_2dp = [100,100,100,100];
                      end
                      fwrite(fid,single(diff_2dt(:)), 'single');
                      fwrite(fid,single(diff_2dp(:)), 'single');
                   end
               else
                   boxes(bi).gtbb =[];
                   fwrite(fid,uint8(0), 'uint8');
               end
                
            end
        end
        
    end
    fclose (fid);
    fprintf('saving to mat...\n')
    if strcmp(train_or_test,'train') 
       save([outpath '/' filename '.mat'],'BoxperImage','totalnumofbox','gt','NYUonly','-v7.3'); 
    else
       save([outpath '/' filename '.mat'],'BoxperImage','filelog','totalnumofbox','gt','NYUonly','-v7.3');
    end
else
    load([outpath '/' filename '.mat'],'totalnumofbox');
    fprintf('return: %s\n',boxfile);
end
%{
if 0
    imagecount = 1;
    fid = fopen('/Users/shuran/Dropbox/marvin/DSS/boxfile/boxes_NYU_po_train_tar.list');

    while 1 
        
        len = fread(fid , 1, 'int32');	
        name = fread(fid , len,  'char*1');	
        name = [char(name')];
        display(name)
        pause;
        if  strcmp(name,'SUNRGBD/kv1/NYUdata/NYU0503')
            break;
        end
        R = fread(fid , 9, 'single');	
        K = fread(fid , 9, 'single');	
        h = fread(fid , 1, 'uint32');	
        w = fread(fid , 1, 'uint32');	
        len = fread(fid , 1, 'uint32');
        clear box;
        for i =1:len
            box(i).cls = fread(fid , 1, 'uint32');
            box(i).basis = reshape(fread(fid , 9, 'single'),3,3);	
            box(i).centroid = fread(fid , 3, 'single');	
            box(i).coeffs = fread(fid , 3, 'single');	

            hasTarget = fread(fid , 1, 'uint8');
            if hasTarget>0
               box(i).gt.basis = reshape(fread(fid , 9, 'single'),3,3);	
               box(i).gt.centroid = fread(fid , 3, 'single');
               box(i).gt.coeffs = fread(fid , 3, 'single');	
            end

        end
        imagecount = imagecount+1;
    end
end
            
        
end


			
%}
