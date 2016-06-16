function outfile = dss_RPNpreparelist(train_or_test,NYUonly)
SUNRGBDtoolboxdir = './external/prepareGT/SUNRGBDtoolbox/';
outpath = '/n/fs/modelnet/deepDetect/code/marvin/DSS/boxfile/';

if NYUonly
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat')
    alltrain = trainSeq;
    alltest = testSeq;
    filename = 'RPN_NYU';
else
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/allsplit.mat')
    filename = 'RPN_SUNrgbd';
end


if strcmp(train_or_test,'train') 
   filename = [filename '_train'];
   allpath = alltrain;
   
elseif strcmp(train_or_test,'test') 
    allpath = alltest;
    filename = [filename '_test'];
    
else strcmp(train_or_test,'train_test')
    allpath = [alltest,alltrain];
    
end
seqnames = cell(1,length(allpath));
for i =1:length(allpath)
    seqnames{i} = getSequenceName(allpath{i});
end
%filename ='0962';
%seqnames = {'SUNRGBD/kv1/NYUdata/NYU0962'};
outfile = [outpath '/' filename '.list'];
if ~exist(outfile,'file')
    fprintf('out: %s\n',outfile);
    load([SUNRGBDtoolboxdir '/Metadata/' 'SUNRGBDMeta.mat']);
    fid = fopen(outfile,'wb');
    totalnumofbox =0;
    for imageNum = 1:length(seqnames)
        fprintf('%d image: %s \n', imageNum,seqnames{imageNum})
        [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
        data = SUNRGBDMeta(ind);
        
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
    end
    
    fclose (fid);
else
    load([outpath '/' filename '.mat'],'totalnumofbox');
    fprintf('return: %s\n',outfile);
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
