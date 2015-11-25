function map =  GetMap(im_sz, fea_sz, roi_size, location, l_off, s, type)
if strcmp(type, 'box')
    map = ones(im_sz);
    map = crop_bg(map, location, [0,0,0]);
elseif strcmp(type, 'gaussian')
    
    map = zeros(im_sz(1), im_sz(2));
    scale = min(location(3:4))/3;
    %     mask = fspecial('gaussian', location(4:-1:3), scale);
    mask = fspecial('gaussian', min(location(3:4))*ones(1,2), scale);
    mask = imresize(mask, location(4:-1:3));
    mask = mask/max(mask(:));
    
    x1 = location(1);
    y1 = location(2);
    x2 = x1+location(3)-1;
    y2 = y1+location(4)-1;
    
    clip = min([x1,y1,im_sz(1)-y2, im_sz(2)-x2]);
    pad = 0;
    if clip<=0
        pad = abs(clip)+1;
        map = zeros(im_sz(1)+2*pad, im_sz(2)+2*pad);
        %         map = padarray(map, [pad, pad]);
        x1 = x1+pad;
        x2 = x2+pad;
        y1 = y1+pad;
        y2 = y2+pad;
    end
    map(y1:y2,x1:x2) = mask;
    if clip<=0
        map = map(pad+1:end-pad, pad+1:end-pad);
    end
else error('unknown map type');
end
map = ext_roi(map, location, l_off, roi_size, s);
map = imresize(map(:,:,1), [fea_sz(1), fea_sz(2)]);
end