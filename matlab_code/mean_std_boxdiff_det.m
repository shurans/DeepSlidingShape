function [means,stds,means_2d_t,stds_2d_t,means_2d_p,stds_2d_p]=mean_std_boxdiff_det(proposal_dir,replace)
mean_std_file = fullfile(proposal_dir,'mean_std.mat');
fprintf('generating %s ...\n',mean_std_file);

load(['/n/fs/modelnet/SUN3DV2/prepareGT/cls.mat']);
if 0 %exist(mean_std_file,'file')&&replace
   load(mean_std_file);
   
else   
    class_counts = zeros(length(cls),1);
    sum_x = zeros(length(cls),6);
    sum_sqr_x = zeros(length(cls),6);
    
    sum_x_2d_t  = zeros(length(cls),4);
    sum_sqr_x_2d_t = zeros(length(cls),4);
    
    sum_x_2d_p  = zeros(length(cls),4);
    sum_sqr_x_2d_p = zeros(length(cls),4);
    
    Allfiles = dirAll (proposal_dir,'.mat') ;
    for i =1:10:length(Allfiles)
        a = load(Allfiles{i});
        if isfield(a,'candidates3d')
            candidates3d = a.candidates3d;
            for bi =1:length(candidates3d) 
                [~,cid] = ismember(candidates3d(bi).classname,cls);
                if cid >0&&candidates3d(bi).iou>0.25&&~isempty(candidates3d(bi).diff_2dt)&&~isempty(candidates3d(bi).diff_2dp)
                   class_counts(cid) = class_counts(cid)+1;
                   sum_x(cid,:) = sum_x(cid,:)+candidates3d(bi).diff;
                   sum_sqr_x(cid,:) = sum_sqr_x(cid,:)+candidates3d(bi).diff.*candidates3d(bi).diff;

                   sum_x_2d_t(cid,:)      = sum_x_2d_t(cid,:) + candidates3d(bi).diff_2dt;
                   sum_sqr_x_2d_t(cid,:)  = sum_sqr_x_2d_t(cid,:) + candidates3d(bi).diff_2dt.*candidates3d(bi).diff_2dt;

                   sum_x_2d_p(cid,:) = sum_x_2d_p(cid,:) + candidates3d(bi).diff_2dp;
                   sum_sqr_x_2d_p(cid,:) = sum_sqr_x_2d_p(cid,:) +  + candidates3d(bi).diff_2dp.*candidates3d(bi).diff_2dp;
                end
            end
        end
    end
    means    = sum_x ./ repmat(class_counts,[1,6]);
    stds     = sqrt(sum_sqr_x ./repmat(class_counts,[1,6]) - means.*means);
    
    means_2d_t = sum_x_2d_t ./ repmat(class_counts,[1,4]);
    stds_2d_t  = sqrt(sum_sqr_x_2d_t ./repmat(class_counts,[1,4]) - means_2d_t.*means_2d_t);
    
    means_2d_p = sum_x_2d_p ./ repmat(class_counts,[1,4]);
    stds_2d_p  = sqrt(sum_sqr_x_2d_p ./repmat(class_counts,[1,4]) - means_2d_p.*means_2d_p);
    
    display(means)
    display(stds)
    save(mean_std_file,'means','stds','means_2d_p','stds_2d_p','means_2d_t','stds_2d_t');
end
end