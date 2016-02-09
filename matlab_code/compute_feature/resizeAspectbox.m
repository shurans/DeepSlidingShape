function boxes = resizeAspectbox(boxes)
        boxes.coeffs(boxes.coeffs<0.15)=0.15;
        [coff_sort,ind_sort] = sort(boxes.coeffs);
        if (coff_sort(1)/coff_sort(3)<0.15)
           scale = min(2,coff_sort(2)/coff_sort(1));           
           boxes.coeffs(ind_sort(1)) = scale*boxes.coeffs(ind_sort(1));
        end
end