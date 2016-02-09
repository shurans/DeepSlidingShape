function getboxdistribution()
    fullbox = 1;
    load('/n/fs/modelnet/SUN3DV2/prepareGT/cls');
    load('/n/fs/modelnet/SUN3DV2/prepareGT/Metadata/SUNRGBDMeta_tight_Yaw.mat');
    if fullbox
        groundtruth = [SUNRGBDMeta.groundtruth3DBB];
    else
        groundtruth = [SUNRGBDMeta.groundtruth3DBB_tight];
    end
    if fullbox
        sizemodel_dir = ['/n/fs/modelnet/deepDetect/code/detector_3d/sizedistrubution/sizedata_full/'];
    else
        sizemodel_dir = ['/n/fs/modelnet/deepDetect/code/detector_3d/sizedistrubution/sizedata_tight/'];
    end
    for i = 1:length(cls)
        classname = cls{i};
        boxes = groundtruth(ismember({groundtruth.classname},classname));
        coeffs = reshape([boxes.coeffs],3,[]);
        coeffsxy = coeffs(1:2,:);
        coeffsxy = sort(coeffsxy,1);
        coeffs(1:2,:) = coeffsxy;
        center = reshape([boxes.centroid],3,[]);
        
        h = fspecial('gaussian',[1,3]);
        numofbox = size(boxes,2);
        
        [x_coeffs_hist,x_coeffs_bin] = hist(coeffs(1,:),30);
        x_coeffs_hist = conv(x_coeffs_hist/numofbox,h,'same');
        [y_coeffs_hist,y_coeffs_bin] = hist(coeffs(2,:),30);
        y_coeffs_hist = conv(y_coeffs_hist/numofbox,h,'same');
        [z_coeffs_hist,z_coeffs_bin] = hist(coeffs(3,:),30);
        z_coeffs_hist = conv(z_coeffs_hist/numofbox,h,'same');
        [height_coeffs_hist,height_coeffs_bin] = hist(center(3,:),30);
        height_coeffs_hist = conv(height_coeffs_hist/numofbox,h,'same');
        
        [asp(1,:),asp_bin(1,:)] = hist(coeffs(1,:)./coeffs(3,:),30);
        [asp(2,:),asp_bin(2,:)] = hist(coeffs(2,:)./coeffs(3,:),30);
        [asp(3,:),asp_bin(3,:)] = hist(coeffs(1,:)./coeffs(2,:),30);
        
        asp = conv2(asp/numofbox,h,'same');
        
        allhist =[x_coeffs_hist;y_coeffs_hist;z_coeffs_hist;height_coeffs_hist;asp];
        allbin = [x_coeffs_bin;y_coeffs_bin;z_coeffs_bin;height_coeffs_bin;asp_bin];
        
        save([sizemodel_dir classname '.mat'],'allhist','allbin');
    end

