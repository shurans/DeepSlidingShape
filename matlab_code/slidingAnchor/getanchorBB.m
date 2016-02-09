%load('/n/fs/modelnet/SUN3DV2/prepareGT/cls');
%load('/n/fs/modelnet/SUN3DV2/prepareGT/Metadata/groundtruth_tight.mat')
load('/Users/shuran/Documents/SUNRGBD/SUNRGBDtoolbox/Metadata/groundtruth_tight.mat')
load('/Users/shuran/Documents/SUNRGBD/SUNRGBDtoolbox/Metadata/detection19list.mat')
cls = detection19list;

boxes = groundtruth(ismember({groundtruth.classname},cls));
coeffs = reshape([boxes.coeffs],3,[]);
coeffsxy = coeffs(1:2,:);
coeffsxy = sort(coeffsxy,1);
coeffs(1:2,:) = coeffsxy;

%{
valid = ones(1,length(coeffs(1,:)));
coeffs_range = zeros(3,2);
pt = 3;
for i =1:3
coeffs_range(i,:) = [prctile(coeffs(i,:),pt),prctile(coeffs(i,:),100-pt)];
valid = valid&coeffs(i,:)>coeffs_range(i,1)&coeffs(i,:)<coeffs_range(i,2);
end

coeffs = coeffs(:,valid);
plot3(coeffs(1,:),coeffs(2,:),coeffs(3,:),'.');
%}
numofcenter = 9;
[idx,center] = kmeans(coeffs',numofcenter);
center = center';
idx = idx';
figure,
plot3(center(1,:),center(2,:),center(3,:),'xr','MarkerSize',20);
hold on;
for i =1:numofcenter
    plot3(coeffs(1,idx==i),coeffs(2,idx==i),coeffs(3,idx==i),'.','MarkerFaceColor',rand(1,3));
end

vol = center(1,:).*center(2,:).*center(3,:);
[~,ind]  = sort(vol);
ratio = center(2,:)./center(1,:);
[ratio,ind]  = sort(ratio);
center = center(:,ind);

R1 = getRotationMatrix('z',0.25*pi); R1 = R1(1:3,1:3);
R2 = getRotationMatrix('z',-0.25*pi);R2 = R2(1:3,1:3);
%%
BaseAll = {eye(3),[0,1,0;1,0,0;0,0,1]};
cnt = 1;
clear anchorBox;
for i =1:length(center)
    if  center(1,i)/center(2,i)==1
        Base = BaseAll(1);
    else 
        Base = BaseAll;
    end
    
    for j = 1:length(Base)
        anchorBox(cnt).basis = Base{j};
        anchorBox(cnt).centroid = [0,0,0];
        anchorBox(cnt).coeffs = center(:,i)';
        cnt = cnt+1;
    end

end

figure,
clf
for i =1:length(anchorBox)
hold on;
vis_cube(anchorBox(i),[acoeffs(3,i)<=0.35 0.1 0]);
pause;
end
anchorBoxAlign = makeaxisAlign(anchorBox);
scoreMatrix = bb3dOverlapCloseForm(anchorBox',anchorBox');
figure,imagesc(scoreMatrix)
save('anchorBox','anchorBoxAlign','anchorBox','center');
Space  = struct('Rx',[-2.5,2.5],'Ry',[1,6],'Rz',[-1.5,1.5],'s',s);

acoeffs = reshape([anchorBox.coeffs],3,[]);
size_group = zeros(1,length(anchorBox));
size_group(acoeffs(3,:)<=0.35) = 1;
size_group = zeros(1,length(anchorBox));
size_group(acoeffs(3,:)<=0.3) = 1;
save('anchorBox','anchorBoxAlign','anchorBox','center');