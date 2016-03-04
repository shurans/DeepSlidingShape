function orgnizefeat(featurefolder,featureLocalfolder,boxfilename,cls,opt)
ind = findstr(boxfilename,'.list');
if ~isempty(ind)
   boxfilename = boxfilename(1:ind(1)-1);
end

fprintf('loading ... %s\n',[boxfilename '.mat']); 
ld = load([boxfilename '.mat']);
totalnumofbox = ld.totalnumofbox;
filelog = ld.filelog;
gt = ld.gt;
BoxperImage = ld.BoxperImage;
clear ld;

if gt
   tosacvefeaturefolder =[featurefolder '/gt/'];
   toreadfeaturefolder =[featureLocalfolder '/gt/'];
else
   tosacvefeaturefolder =[featurefolder '/po/'];
   toreadfeaturefolder  = [featureLocalfolder '/po/'];
end


if exist([toreadfeaturefolder 'fc5.tensor'],'file')&&opt.loadfea
    opt.loadfea = 1;
    fprintf('loading ... %s\n',[toreadfeaturefolder 'fc5.tensor']); 
    features_fc5 = readTensor([toreadfeaturefolder 'fc5.tensor'],true);
    features_fc5 = features_fc5.value;
    features_fc5 = squeeze(features_fc5);
    features_fc5 = features_fc5';
else
    opt.loadfea  = 0;
end



fprintf('loading ... %s\n',[toreadfeaturefolder 'fc6.tensor']); 
features_fc6 = readTensor([toreadfeaturefolder 'fc6.tensor'],true);
features_fc6 = features_fc6.value;
features_fc6 = squeeze(features_fc6);
features_fc6 = features_fc6';
[~,class_res] = max(features_fc6,[],2);
class_res = class_res'-1;


if opt.box_reg
   fprintf('loading ... %s\n',[toreadfeaturefolder 'box_pred.tensor']); 
   box_pred = readTensor([toreadfeaturefolder 'box_pred.tensor'],true);
   box_pred = box_pred.value;
   box_pred = squeeze(box_pred);
   box_pred = box_pred';
end

if opt.orein_cls
   fprintf('loading ... %s\n',[toreadfeaturefolder 'oreintation.tensor']); 
   oreintation = readTensor([toreadfeaturefolder 'oreintation.tensor'],true);
   oreintation = oreintation.value;
   oreintation = squeeze(oreintation);
   oreintation = oreintation';
end

start = 1;
classifyAcc = 0;
for imagNum =1:length(filelog)
    % load box 
    if gt
        boxes = filelog(imagNum).boxesfile;
    else
        ld = load(filelog(imagNum).boxesfile);
        boxes = ld.candidates3d;
        clear ld;%boxes = filelog(imagNum).boxes;
        if BoxperImage>0
           boxes = boxes(1:min(BoxperImage,length(boxes)));
        end
    end
            
    f.boxes = processbox(boxes);
    f.seqname = filelog(imagNum).seqname;
    numofbox = length(boxes);
    
    if opt.loadfea
        f.feat  = features_fc5(start:start+numofbox-1,:);
    end
    f.score = features_fc6(start:start+numofbox-1,:);
    f.class_res = class_res(start:start+numofbox-1);
    if opt.box_reg
       f.box_pred = box_pred(start:start+numofbox-1,:);
    end
    
    if opt.orein_cls 
       o_pred = oreintation(start:start+numofbox-1,:);
       for i = 1:numofbox
           f.boxes(i).o_pred = o_pred(i,:);
       end
    end
    
    if BoxperImage>0
       start = start+BoxperImage;
    else
       start = start+numofbox;
    end
    
    if numofbox>0
        if isfield(f.boxes,'classname')
            [~,cid]=ismember({f.boxes.classname},cls);
        elseif isfield(f.boxes,'labelname')
            [~,cid]=ismember({f.boxes.labelname},cls);
        else
            cid = 0;
        end
        
        if isfield(f.boxes,'iou')
           cid([f.boxes.iou]<0.25) = 0;
        end
        classifyAcc = classifyAcc + sum(cid==f.class_res);
    end
    
    tosavename = fullfile(tosacvefeaturefolder,[filelog(imagNum).seqname '.mat']);
    ind = find(tosavename=='/');
    if ~exist(tosavename(1:ind(end)),'dir'),mkdir(tosavename(1:ind(end)));end
    if ~exist(tosavename,'file')
        save(tosavename,'f');
        fprintf('saving: %s\n',tosavename);
    else
        fprintf('skip: %s\n',tosavename);
    end
end
fprintf('classifyAcc =%f\n',classifyAcc/totalnumofbox);

end