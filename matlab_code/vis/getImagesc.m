function colorImg = getImagesc(img,fix,c)
        if ~exist('c','var')
            nColors = 64;
            c = jet(nColors);
        end
    
    if exist('fix','var')&&~isempty(fix),   
         img(isnan(img)) = fix(1);
         img(img>fix(2))=fix(2);
         img = (img - fix(1)) / (fix(2)- fix(1));
         img = ceil(img * nColors);
         img(img<=0) = 1;
    else
        if isnan(nanmin(img(:)))||nanmax(img(:))==nanmin(img(:)),
            img =ones(size(img));
        else
            img(isnan(img))=nanmin(img(:))-0.1*(nanmax(img(:))- nanmin(img(:)));
            img = (img - nanmin(img(:))) / (nanmax(img(:))- nanmin(img(:)));
            img = ceil(img * size(c,1));
            img(img<=0) = 1;
        end  
    end
    colorImg = zeros([size(img) 3]);
    colorImg(:,:,1) = reshape(c(img,1),size(img));
    colorImg(:,:,2) = reshape(c(img,2),size(img));
    colorImg(:,:,3) = reshape(c(img,3),size(img));
end