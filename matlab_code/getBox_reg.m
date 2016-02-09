function boxes = getBox_reg(d_strcut,class_id,mean_std)
         num_param = 6;
         
         boxes = d_strcut.boxes;
         m = mean_std.means(class_id,:);
         s = mean_std.stds(class_id,:);
         
         idx = num_param*(class_id)+1;
         box_pred = d_strcut.box_pred(:,idx :idx+5);
         box_pred = bsxfun(@times,box_pred,s);    
         box_pred = bsxfun(@plus,box_pred,m);    
         for boxid = 1:length(boxes)
             boxes(boxid,:).centroid = d_strcut.boxes(boxid,:).centroid + box_pred(boxid,1:3);
             boxes(boxid,:).coeffs   = d_strcut.boxes(boxid,:).coeffs   + box_pred(boxid,4:6);
         end         
end