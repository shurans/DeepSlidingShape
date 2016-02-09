function bbtight3d = transform2tightBB(bbw_3d_d,BBTight,Rot)
if isempty(bbw_3d_d) 
    bbtight3d =struct([]);
    return;
end
nBB =size(bbw_3d_d,1);
centroid  =bbw_3d_d(:,1:3)+ 0.5*bbw_3d_d(:,4:6);
centroid = mat2cell(centroid,ones(1,nBB),3);
coeffs =mat2cell(repmat(BBTight.coeffs,[nBB,1]),ones(1,nBB),3);
basisR = BBTight.basis;
if ~isempty(Rot)
   basisR = [Rot(1:3,1:3)'*basisR']';
end
basis = mat2cell(repmat(basisR,[nBB,1]),3*ones(1,nBB),3);

if size(bbw_3d_d,2)==7
    confidence = num2cell(bbw_3d_d(:,7));
    bbtight3d =struct('centroid',centroid,'basis',basis,'coeffs',coeffs,'confidence',confidence);
else
    bbtight3d =struct('centroid',centroid,'basis',basis,'coeffs',coeffs);   
end
end
