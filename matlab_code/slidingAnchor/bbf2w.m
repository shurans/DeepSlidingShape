function bbw =bbf2w(bb_in,Space)
bbw =bb_in;
if ~isempty(bb_in)
    bbw(:,1:3) =(bb_in(:,1:3)-1)*Space.s...
               +repmat([Space.Rx(1) Space.Ry(1) Space.Rz(1)],[size(bb_in,1),1]);
    bbw(:,4:6) = (bb_in(:,4:6)+1)*Space.s;
end
end
