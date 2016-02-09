function compute_targetboxdiff()
% cd /n/fs/modelnet/deepDetect/code/detector_3d/marvin
% load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/allsplit.mat');
% allPath = [alltest,alltrain];
load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/test_kv1NYU.mat')
load('/n/fs/modelnet/SUN3DV2/prepareGT/traintestSUNRGBD/train_kv1NYU.mat');
load(['/n/fs/modelnet/SUN3DV2/prepareGT/cls.mat']);
allPath =  [testSeq,trainSeq];

proposal_dir_update = '/n/fs/modelnet/deepDetect/proposal/rgbd_tight_2/';
SUNRGBDtoolboxdir = '/n/fs/modelnet/SUN3DV2/prepareGT/';
load([SUNRGBDtoolboxdir '/Metadata/' 'SUNRGBDMeta_tight.mat']);
seqnames = cell(1,length(allPath));
for i =1:length(allPath)
    seqnames{i} = getSequenceName(allPath{i});
end

if ~exist('id','var')
    imageNums = 1:length(allPath);
else
    imageNums =id:300:length(allPath);
end
class_counts = zeros(length(cls),1);
sum_x = zeros(length(cls),6);
sum_sqr_x = zeros(length(cls),6);

for imageNum = imageNums
    fprintf('%d image: %s \n', imageNum,seqnames{imageNum})
    [~,ind]=ismember(seqnames{imageNum},{SUNRGBDMeta.sequenceName});
    data = SUNRGBDMeta(ind);
    tosavepath = fullfile(proposal_dir_update,data.sequenceName);
    load(tosavepath,'candidates3d','eval');
    % change oreitation of candidates3d
    candidates3d = processbox(candidates3d);
    % find the match between gt and candidates3d
    for bi =1:length(candidates3d) 
        % caculate diff 
        if ~isempty(candidates3d(bi).gtbb)
            candidates3d(bi).gtbb =  processbox(candidates3d(bi).gtbb);
            dotproduct = sum(repmat(candidates3d(bi).basis(1,:),3,1).*candidates3d(bi).gtbb.basis,2);
            [~,ind] = sort(dotproduct(1:2),'descend');
            candidates3d(bi).diff(1:3) = candidates3d(bi).gtbb.centroid -candidates3d(bi).centroid;
            candidates3d(bi).diff(4:6) = candidates3d(bi).gtbb.coeffs([ind;3]) - candidates3d(bi).coeffs;
            % visulize box + diff
            regressbb = candidates3d(bi);
            regressbb.centroid = regressbb.centroid+candidates3d(bi).diff(1:3);
            regressbb.coeffs = regressbb.coeffs+candidates3d(bi).diff(4:6);

            % save the values for std and mean 
            [~,cid] = ismember(candidates3d(bi).classname,cls);
            if cid >0&&candidates3d(bi).iou>0.25
               class_counts(cid) = class_counts(cid)+1;
               sum_x(cid,:) = sum_x(cid,:)+candidates3d(bi).diff;
               sum_sqr_x(cid,:) = sum_sqr_x(cid,:)+candidates3d(bi).diff.*candidates3d(bi).diff;
               %{
               clf
               vis_cube(candidates3d(bi),'r');hold on;
               vis_cube(candidates3d(bi).gtbb,'g');
               vis_cube(regressbb,'b');
               %pause;
               %}
            end
        end
        
    end
   
    save(tosavepath,'candidates3d','eval');
end
% var(x) = E(x^2) - E(x)^2
means = sum_x ./ repmat(class_counts,[1,6]);
stds =  sqrt(sum_sqr_x ./repmat(class_counts,[1,6]) - means.*means);
display(means)
display(stds)
save(fullfile(proposal_dir_update,'mean_std'),'means','stds');

end