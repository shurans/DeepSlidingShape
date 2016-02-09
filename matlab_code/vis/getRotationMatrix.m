function R = getRotationMatrix(axis,theta)
    switch axis
        case 'x',
            R = [1,           0,           0,           0;
                 0,           cos(theta),  -sin(theta), 0;
                 0,           sin(theta),   cos(theta), 0;
                 0,           0,           0,           1];
        case 'y',
            R = [cos(theta),  0,  sin(theta),           0;
                 0,           1,           0,           0;
                 -sin(theta), 0,  cos(theta),           0;
                 0,           0,           0,           1];
        case 'z',
            R = [cos(theta),  -sin(theta), 0,           0;
                 sin(theta),   cos(theta), 0,           0;
                 0,           0,           1,           0;
                 0,           0,           0,           1];
        case 'tilt',
            R = [1,           0,           0,           0;
                 0,           cos(theta),  -sin(theta), 0;
                 0,           sin(theta),   cos(theta), 0;
                 0,           0,           0,           1];
             R = R(1:3,1:3);
        otherwise,
            error('Invalid axis. Must be "x" or "y" or "z" or "tilt".');
    end
end