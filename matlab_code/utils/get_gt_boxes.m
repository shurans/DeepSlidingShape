function gt_boxes = get_gt_boxes(groundtruthBB,seqname, class)
id1 = ismember({groundtruthBB.sequenceName},seqname);
id2 = ismember({groundtruthBB.classname}, class);
gt_boxes = groundtruthBB(id1&id2);
end