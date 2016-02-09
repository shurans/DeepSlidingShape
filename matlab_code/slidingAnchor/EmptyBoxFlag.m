function [EmptyBox,CountBox] = EmptyBoxFlag(pointCountIntegral,sizeTemplate,thr)
sizePtI = size(pointCountIntegral);
% EmptyBox = pointCountIntegral(sizeTemplate(1):end,sizeTemplate(2):end,sizeTemplate(3):end)-...
%            pointCountIntegral(1:sizePtI(1)-sizeTemplate(1)+1,1:sizePtI(2)-sizeTemplate(2)+1,1:sizePtI(3)-sizeTemplate(3)+1);
y1 = 1:sizePtI(1)-sizeTemplate(1);
y2 = sizeTemplate(1)+1:sizePtI(1);
x1 = 1:sizePtI(2)-sizeTemplate(2);
x2 = sizeTemplate(2)+1:sizePtI(2);
z1 = 1:sizePtI(3)-sizeTemplate(3);
z2 = sizeTemplate(3)+1:sizePtI(3);

CountBox = pointCountIntegral(y2,x2,z2) - pointCountIntegral(y1,x2,z2) - pointCountIntegral(y2,x1,z2) - pointCountIntegral(y2,x2,z1) + pointCountIntegral(y2,x1,z1) + pointCountIntegral(y1,x2,z1) + pointCountIntegral(y1,x1,z2) - pointCountIntegral(y1,x1,z1);
EmptyBox =CountBox<thr;
end