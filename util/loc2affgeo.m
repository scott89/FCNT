function param = loc2affgeo(location, p_sz)
% location = [tlx, tly, w, h]

cx = location(1)+(location(3)-1)/2;
cy = location(2)+(location(4)-1)/2;
param = [cx, cy, location(3)/p_sz, 0, location(4)/location(3), 0]';

end