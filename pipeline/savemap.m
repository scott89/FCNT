l_map = imresize(l_pre_map, [roi_size, roi_size]);
l_map = grs2rgb(mat2gray(l_map), jet);
g_map = imresize(g_pre_map, [roi_size, roi_size]);
g_map = grs2rgb(mat2gray(g_map), jet);
[roi, roi_pos, padded_zero_map, pad] = ext_roi(im2, location, l2_off,  roi_size, s2);
imwrite(mat2gray(roi), sprintf('figure1/roi_%04d.png',im2_id))
imwrite(l_map, sprintf('figure1/l_%04d.png',im2_id));
imwrite(g_map, sprintf('figure1/g_%04d.png',im2_id));

% gfea = fea2{2};
% gfea = permute(gfea, [2,1,3]);
% gfea = imresize(double(gfea), [roi_size, roi_size]);
% lfea = fea2{1};
% lfea = permute(lfea, [2,1,3]);
% lfea = imresize(double(lfea), [roi_size, roi_size]);
% 
% for i=1:size(gfea, 3)
%     imwrite(grs2rgb(mat2gray(gfea(:,:,i))+0.0001, hot), sprintf('pipeline/gfea_%04d.png', i));
%     imwrite(grs2rgb(mat2gray(lfea(:,:,i))+0.0001, hot), sprintf('pipeline/lfea_%04d.png', i));
% end
%     