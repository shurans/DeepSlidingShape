% change
load(['SUNRGBDMeta.mat']);
change = zeros(1,length(SUNRGBDMeta));
for i =1:length(SUNRGBDMeta)
    dir = SUNRGBDMeta(i).Rtilt*[0;1;0];
    angle = dir(1:2,1)/norm(dir(1:2,1));
    if (angle(2)<0.9)
        change(i) = 1;
        [rgb,points3d,imgZ]=read3dPoints(SUNRGBDMeta(i));
        dir = SUNRGBDMeta(i).Rtilt*[0;1;0];
        angle = dir(1:2,1)/norm(dir(1:2,1));
        if (angle(2)<0.9)
            change(i) = 2;            
            R = getRotationMatrix('z',atan2(angle(1),angle(2)));
            R = R(1:3,1:3);
            SUNRGBDMeta(i).Rtilt = R*SUNRGBDMeta(i).Rtilt;
            for k =1:length(SUNRGBDMeta(i).groundtruth3DBB)
                SUNRGBDMeta(i).groundtruth3DBB(k).centroid = [R*SUNRGBDMeta(i).groundtruth3DBB(k).centroid']';
                SUNRGBDMeta(i).groundtruth3DBB(k).basis = [R*SUNRGBDMeta(i).groundtruth3DBB(k).basis']';
            end
        end
    end 
end
