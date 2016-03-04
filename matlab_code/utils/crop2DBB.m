function allBb2d = crop2DBB(allBb2d,imh,imw)
        if isempty(allBb2d)
            return;
        end
        allBb2d_comp1 = allBb2d(:,[1,2]);
        allBb2d_comp1(allBb2d_comp1<1) =1;
        allBb2d_comp3 = allBb2d(:,1)+allBb2d(:,3);
        allBb2d_comp3(allBb2d_comp3>imw)=imw;
        allBb2d_comp4 = allBb2d(:,2)+allBb2d(:,4);
        allBb2d_comp4(allBb2d_comp4>imh)=imh;
        if size(allBb2d,2)>4
            allBb2d = [allBb2d_comp1,allBb2d_comp3,allBb2d_comp4,allBb2d(:,5:end)];
        else
            allBb2d = [allBb2d_comp1,allBb2d_comp3,allBb2d_comp4];
        end
        allBb2d(:,[3,4])= allBb2d(:,[3,4])-allBb2d(:,[1,2]);
        allBb2d(allBb2d(:,3)<0| allBb2d(:,4)<0,:)=[];
end