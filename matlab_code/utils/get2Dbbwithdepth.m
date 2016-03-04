function [gtBb2D,label,gtBb2Dproject] = get2Dbbwithdepth(gtBB,imsize,points3d,depthInpaint,data)
        label =1;
        if ~isfield(gtBB,'gtBb2D')
           [bb2d,bb2dDraw] = projectStructBbsTo2d(gtBB,data.Rtilt,[],data.K);
           groundtruth2Dfrom3DBB = bb2d(1:4);
        elseif sum(isnan(gtBB.gtBb2D(1:4)))>0,
            %gtBB.gtBb2D = [];
            groundtruth2Dfrom3DBB = [];
        else
            groundtruth2Dfrom3DBB = gtBB.gtBb2D(1:4);
            %groundtruth2Dfrom3DBB = crop2DBB(groundtruth2Dfrom3DBB,imsize(1),imsize(2));
        end

        gtBb2Dproject = groundtruth2Dfrom3DBB;
        isIntightBBall = ptsInTightBB(points3d,gtBB);
        if sum(isIntightBBall)>min(1000,1000*prod(gtBB.coeffs))
            mask = zeros(imsize);
            mask(find(isIntightBBall)) = 1;
            gtBb2D =findCropRegion(mask);
        else
            gtBb2D= [];
        end
       
        
%         if ~isempty(groundtruth2Dfrom3DBB),
%             groundtruth2Dfrom3DBB = round(groundtruth2Dfrom3DBB);
%             depthinside = depthInpaint(groundtruth2Dfrom3DBB(2):groundtruth2Dfrom3DBB(2)+groundtruth2Dfrom3DBB(4),...
%                                        groundtruth2Dfrom3DBB(1):groundtruth2Dfrom3DBB(1)+groundtruth2Dfrom3DBB(3));
%             missingdepthRatio = sum(depthinside(:) ==0)/length(depthinside(:));     
%             if  missingdepthRatio>0.4,
%                 % if too many missing data mark it
%                 label =2;
%             else 
%                 label =1;
%             end
%         else
%             label =2;
%         end

        if isempty(gtBb2D)||prod(gtBb2D([3,4]))<25
            gtBb2D =[];
            label =2;
        else
%             if ~isempty(gtBb2Dproject)&&(prod(gtBb2D([3,4]))<100||prod(gtBb2D([3,4]))/prod(gtBb2Dproject([3,4]))<0.25)
%                gtBb2D = 0.7*gtBb2D+0.3*gtBb2Dproject;
%             end
        end
        %groundtruth2Dfrom3DBB = crop2DBB(groundtruth2Dfrom3DBB,imsize(1),imsize(2));
        gtBb2D = crop2DBB(gtBb2D,imsize(1),imsize(2));

end