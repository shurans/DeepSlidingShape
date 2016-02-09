function to_remove = removeBoxsizedis(boxes,sizeModel)
        threshold = 0.001;
        coeffs = reshape([boxes.coeffs],3,[]);
        coeffsxy = coeffs(1:2,:);
        coeffsxy = sort(coeffsxy,1);
        coeffs(1:2,:) = coeffsxy;
        
        center = reshape([boxes.centroid],3,[]);
        height = center(3,:);
        
        asp(1,:) = coeffs(1,:)./coeffs(3,:);
        asp(2,:) = coeffs(2,:)./coeffs(3,:);
        asp(3,:) = coeffs(1,:)./coeffs(2,:);
        
        boxvalue = [coeffs;height;asp];
        numofref = size(sizeModel.allhist,1);
        numofbin = size(sizeModel.allhist,2);
        numofcls = size(sizeModel.allbin,3);
        to_remove = zeros(length(boxes),numofcls);
        
        for boxid = 1:length(boxes)
            for clsid = 1:numofcls
                value = boxvalue(:,boxid);
                diff = abs(repmat(value,[1,numofbin]) - sizeModel.allbin(:,:,clsid));
                [mindiff,binidx] = min(diff,[],2);
                dis = sizeModel.allhist(sub2ind(size(sizeModel.allhist),1:numofref,binidx',clsid*ones(1,numofref)));
                dis(mindiff > 1.5*(sizeModel.allbin(:,2,clsid) - sizeModel.allbin(:,1,clsid))) = 0;
                
                to_remove(boxid,clsid) = any(dis<threshold);
            end
        end
        %display(to_remove);
        %pause;
end