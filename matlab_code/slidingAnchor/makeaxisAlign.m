function boxOut = makeaxisAlign(boxIn)
        boxOut = boxIn;
        for i =1:length(boxIn)
            corners = get_corners_of_bb3d(boxIn(i));
            bb2(1) = min(corners(:,1));
            bb2(2) = min(corners(:,2));
            bb2(3) = min(corners(:,3));
            bb2(4) = max(corners(:,1));
            bb2(5) = max(corners(:,2));
            bb2(6) = max(corners(:,3));
            bb3d = [bb2(1:3),bb2(4:6)-bb2(1:3)];
            
            boxOut(i).basis = eye(3);
            boxOut(i).centroid = [bb3d(:,1:3) + 0.5*bb3d(:,4:6)];
            boxOut(i).coeffs = [0.5*bb3d(:,4:6)];

        end

         

end

%
% Args:
%   bb3d - 3D bounding box struct.
%
% Returns:
%   corners - 8x3 matrix of 3D coordinates.
%
% See:
%   create_bounding_box_3d.m
%
% Author: Nathan Silberman (silberman@cs.nyu.edu)
function corners = get_corners_of_bb3d(bb3d)
  corners = zeros(8, 3);
  
  % Order the bases.
  [~, inds] = sort(abs(bb3d.basis(:,1)), 'descend');
  basis = bb3d.basis(inds, :);
  coeffs = bb3d.coeffs(inds);
  
  [~, inds] = sort(abs(basis(2:3,2)), 'descend');
  if inds(1) == 2
    basis(2:3,:) = flipdim(basis(2:3,:), 1);
    coeffs(2:3) = flipdim(coeffs(2:3), 2);
  end
  
  % Now, we know the basis vectors are orders X, Y, Z. Next, flip the basis
  % vectors towards the viewer.
  basis = flip_towards_viewer(basis, repmat(bb3d.centroid, [3 1]));
  
  coeffs = abs(coeffs);

  corners(1,:) = -basis(1,:) * coeffs(1) + basis(2,:) * coeffs(2) + basis(3,:) * coeffs(3);
  corners(2,:) =  basis(1,:) * coeffs(1) + basis(2,:) * coeffs(2) + basis(3,:) * coeffs(3);
  corners(3,:) =  basis(1,:) * coeffs(1) + -basis(2,:) * coeffs(2) + basis(3,:) * coeffs(3);
  corners(4,:) = -basis(1,:) * coeffs(1) + -basis(2,:) * coeffs(2) + basis(3,:) * coeffs(3);
  
  corners(5,:) = -basis(1,:) * coeffs(1) + basis(2,:) * coeffs(2) + -basis(3,:) * coeffs(3);
  corners(6,:) =  basis(1,:) * coeffs(1) + basis(2,:) * coeffs(2) + -basis(3,:) * coeffs(3);
  corners(7,:) =  basis(1,:) * coeffs(1) + -basis(2,:) * coeffs(2) + -basis(3,:) * coeffs(3);
  corners(8,:) = -basis(1,:) * coeffs(1) + -basis(2,:) * coeffs(2) + -basis(3,:) * coeffs(3);
  
  corners = corners + repmat(bb3d.centroid, [8 1]);
end

function normals = flip_towards_viewer(normals, points)
  points = points ./ repmat(sqrt(sum(points.^2, 2)), [1, 3]);
  
  proj = sum(points .* normals, 2);
  
  flip = proj > 0;
  normals(flip, :) = -normals(flip, :);
end
