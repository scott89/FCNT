load location
im2_name = sprintf([data_path 'img/%04d.jpg'], im2_id);
im2 = double(imread(im2_name));
l_off = location_last(1:2)-location(1:2);
map = GetMap(size(im2), fea_sz, roi_size, location, l_off, s2, 'gaussian');

[roi2, roi_pos, padded_zero_map, pad] = ext_roi(im2, location, l_off,  roi_size, s2);

map = mat2gray(imresize(map, [roi_size, roi_size]));
map = grs2rgb(floor(map*255),jet);

lmap = mat2gray(imresize(l_pre_map, [roi_size, roi_size]));
lmap = grs2rgb(floor(lmap*255),jet);

gmap = mat2gray(imresize(g_pre_map, [roi_size, roi_size]));
gmap = grs2rgb(floor(gmap*255),jet);

imwrite(mat2gray(roi2), sprintf('figure1/%s_roi_%04d.png',set_name,im2_id));
imwrite(map, sprintf('figure1/%s_map_%04d.png',set_name,im2_id));
imwrite(lmap, sprintf('figure1/%s_lmap_%04d.png',set_name,im2_id));
imwrite(gmap, sprintf('figure1/%s_gmap_%04d.png',set_name,im2_id));
%% ================================================================
% map = GetMap(size(im1), fea_sz, roi_size, location, l1_off, s1, 'box');
% map = mat2gray(imresize(map, [roi_size, roi_size]));
% map = grs2rgb(floor(map*255),hot);
% 
% [roi2, roi_pos, padded_zero_map, pad] = ext_roi(im1, location, [0,0],  roi_size, s2);
% 
% 
% lfea = permute(lfea1, [2,1,3]);
% lfea512 = sum(lfea,3);
% lfea512 = mat2gray(imresize(lfea512, [roi_size, roi_size]));
% lfea512 = grs2rgb(floor(lfea512*255),hot);
% 
% lfea384 = sum(lfea(:,:,lid(1:2)),3);
% lfea384 = mat2gray(imresize(lfea384, [roi_size, roi_size]));
% lfea384 = grs2rgb(floor(lfea384*255),hot);
% 
% 
% 
% gfea = permute(gfea1, [2,1,3]);
% gfea512 = sum(gfea,3);
% gfea512 = mat2gray(imresize(gfea512, [roi_size, roi_size]));
% gfea512 = grs2rgb(floor(gfea512*255),hot);
% 
% gfea384 = sum(gfea(:,:,gid(1:10)),3);
% gfea384 = mat2gray(imresize(gfea384, [roi_size, roi_size]));
% gfea384 = grs2rgb(floor(gfea384*255),hot);
% 
% imwrite(mat2gray(roi2(75:end,:,:)), sprintf('figure1/%s_roi_%04d.png',set_name,im1_id));
% imwrite(map(75:end,:,:), sprintf('figure1/%s_map_%04d.png',set_name,im1_id));
% imwrite(gfea512(75:end,:,:), sprintf('figure1/%s_gfea512_%04d.png',set_name,im1_id));
% imwrite(gfea384(75:end,:,:), sprintf('figure1/%s_gfea384_%04d.png',set_name,im1_id));
% imwrite(lfea512(75:end,:,:), sprintf('figure1/%s_lfea512_%04d.png',set_name,im1_id));
% imwrite(lfea384(75:end,:,:), sprintf('figure1/%s_lfea384_%04d.png',set_name,im1_id));
