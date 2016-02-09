function diff = get_regdiff_detect(candidates3d,gtbb)
% call processbox before this function 
% gtbb =  processbox(gtbb); 
% candidates3d =  processbox(candidates3d); 
dotproduct = sum(repmat(candidates3d.basis(1,:),3,1).*gtbb.basis,2);
diff(1:3) = gtbb.centroid -candidates3d.centroid;
[~,ind] = sort(dotproduct(1:2),'descend');
diff(4:6) = gtbb.coeffs([ind;3]) - candidates3d.coeffs;
end