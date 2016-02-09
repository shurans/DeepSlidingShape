function [outfilename,outnumofbox] = genboxesfile_exctractfea(gt,NYUonly,proposal_dir,cls,rescaleBox)
dataroot = '/n/fs/sun3d/data/';
SUNRGBDtoolboxdir = '/n/fs/modelnet/SUN3DV2/prepareGT/';


if NYUonly
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat')
    allpath = [trainSeq,testSeq];
    filenamebase = 'boxes_NYU';
else
    load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/allsplit.mat')
    allpath = [alltrain,alltest];
    filenamebase = 'boxes_SUNrgbd';
end

if gt 
   outpath = '/n/fs/modelnet/deepDetect/proposal/gt/';
   filenamebase = [filenamebase '_gt'];
else
   outpath = proposal_dir;
end

if rescaleBox
   filenamebase = [filenamebase '_rsbox' num2str(rescaleBox)];
end


seqnames = cell(1,length(allpath));
for i =1:length(allpath)
    seqnames{i} = getSequenceName(allpath{i});
end

%% start 
trunckSize = 20000;
numofsplit = 1;%ceil(length(seqnames)/trunckSize);
for splitid = 1:numofsplit
    startInd = (splitid-1)*trunckSize+1;
    endInd = min(length(seqnames),(splitid)*trunckSize);
    filename = [filenamebase num2str(startInd) '_' num2str(endInd)];
    if ~exist([outpath '/' filename '.mat'],'file')
        if ~exist('SUNRGBDMeta','var')
            load([SUNRGBDtoolboxdir '/Metadata/' 'SUNRGBDMeta_tight.mat']);
        end
        fid = fopen([outpath '/' filename '.txt'],'w');
        fprintf('out: %s\n',[outpath '/' filename '.txt'])

        filelog =[];
        totalnumofbox =0;

        for imageNum = startInd:endInd
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
            if gt
                for bi =1:length(data.groundtruth3DBB_tight)
                    if ~isempty(data.groundtruth3DBB_tight(bi))&&ismember(data.groundtruth3DBB_tight(bi).classname,cls)
                        numbox = numbox+1;
                        boxes(numbox) = data.groundtruth3DBB_tight(bi);
                    end
                end
                if numbox>0
                boxes = rmfield(boxes,{'sequenceName','orientation','gtBb2D'});
                end
            else
                ld = load([fullfile(proposal_dir,data.sequenceName) '.mat']);
                boxes = ld.candidates3d;
                numbox = length(boxes);
            end
            fprintf(fid,'%d\n',numbox);

            for bi =1:numbox
                if isfield(boxes(bi),'classname')
                    [~,classid]= ismember(boxes(bi).classname,cls);
                else
                    [~,classid]= ismember(boxes(bi).labelname,cls);
                end
                overlap = 1;% hack to make the cnn work in order, all boxes are forground 

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

            if exist('boxes','var')&&~isempty(boxes)
               filelog(imageNum).boxes = boxes;
            else
               filelog(imageNum).boxes =[]; 
            end
            filelog(imageNum).seqname = seqnames{imageNum};
        end
        fclose (fid);
        save([outpath '/' filename '.mat'],'filelog','totalnumofbox','gt','NYUonly','-v7.3');
    else
        load([outpath '/' filename '.mat'],'totalnumofbox');
    end
    outfilename{splitid} = [outpath '/' filename];
    outnumofbox(splitid) = totalnumofbox;
end
fprintf('out: %s\n',[outpath '/' filename '.txt']);
fprintf('out: %s\n',[outpath '/' filename '.mat']);

end
