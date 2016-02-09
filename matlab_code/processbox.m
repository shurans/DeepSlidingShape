function boxesout = processbox(boxes)
        boxesout = boxes;
        for i = 1:length(boxes)
            if boxesout(i).basis(1,1)<0
               boxesout(i).basis(1,:) = -1*boxes(i).basis(1,:);
            end
            if boxesout(i).basis(2,1)<0
               boxesout(i).basis(2,:) = -1*boxes(i).basis(2,:);
            end
            if abs(boxesout(i).basis(1,1))<abs(boxesout(i).basis(2,1))
               boxesout(i).basis([1,2],:) =  boxesout(i).basis([2,1],:);
               boxesout(i).coeffs([1,2]) =   boxesout(i).coeffs([2,1]);
            end 
        end
end