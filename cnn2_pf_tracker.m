function  cnn2_pf_tracker(set_name, im1_id, ch_num)

set_tracker_param;
caffe('presolve_gnet');
caffe('presolve_lnet');
%% read images
im1_name = sprintf([data_path 'img/%04d.jpg'], im1_id);
im1 = double(imread(im1_name));
if size(im1,3)~=3
    im1(:,:,2) = im1(:,:,1);
    im1(:,:,3) = im1(:,:,1);
end

%% extract roi and display
roi1 = ext_roi(im1, location, l1_off,  roi_size, s1);
%% save roi images
%% ------------------------------
figure(1)
imshow(mat2gray(roi1));
%% preprocess roi
roi1 = impreprocess(roi1);
fea1 = caffe('forward', {single(roi1)});
% ch_num = size(fea1,3);
% ch_num = 128;
fea_sz = size(fea1{1});
lfea1 = fea1{1};
gfea1 = imresize(fea1{2}, fea_sz(1:2));

%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
max_iter_select = 100;
max_iter = 50;
map1 =  GetMap(size(im1), fea_sz, roi_size, location, l1_off, s1, 'gaussian');
map_select = GetMap(size(im1), fea_sz, roi_size, location, l1_off, s1/1.1, 'box');
[lch_w] = GetWeights(permute(map_select,[2,1,3]), lfea1);
[~, lidx] = sort(lch_w, 'descend');
lidx = lidx(1:ch_num);
lidx = 1:512;
[gch_w] = GetWeights(permute(map_select,[2,1,3]), gfea1);
[~, gidx] = sort(gch_w, 'descend');
gidx = gidx(1:ch_num);
gidx = 1:512;

% lfea1 = lfea1(:,:,lidx);
% gfea1 = gfea1(:,:,gidx);

conf_store = 0;
%% select
caffe('set_phase_train');
for i=1:max_iter_select
l_pre_map = caffe('forward_lnet', {lfea1});
l_pre_map = l_pre_map{1};
figure(1011); subplot(1,2,1); imagesc(permute(l_pre_map,[2,1,3]));
l_diff = l_pre_map-permute(single(map1), [2,1,3]);
caffe('backward_lnet', {single(l_diff)});
caffe('update_lnet');

g_pre_map = caffe('forward_gnet', {gfea1});
g_pre_map = g_pre_map{1};
figure(1011); subplot(1,2,2); imagesc(permute(g_pre_map,[2,1,3]));
g_diff = g_pre_map-permute(single(map1), [2,1,3]);
caffe('backward_gnet', {single(g_diff)});
caffe('update_gnet');
fprintf('Iteration %03d/%03d, Local Loss %f, Global Loss %f\n', i, max_iter, sum(abs(l_diff(:))), sum(abs(g_diff(:))));
end


[lsal, lid] = compute_saliency({lfea1}, map1, 'lsolver');
[gsal, gid] = compute_saliency({gfea1}, map1, 'gsolver');

lid = lid(1:ch_num);
gid = gid(1:ch_num);

lfea1 = lfea1(:,:,lid);
gfea1 = gfea1(:,:,gid);
fea2_store = lfea1;
map2_store = map1;
l_distractor_store = false;
%% train
caffe('init_lsolver', lnet_solver_def_file, model_file);
caffe('init_gsolver', gnet_solver_def_file, model_file);
caffe('set_phase_train');
caffe('presolve_gnet');
caffe('presolve_lnet');
for i=1:max_iter
l_pre_map = caffe('forward_lnet', {lfea1});
l_pre_map = l_pre_map{1};
figure(1011); subplot(1,2,1); imagesc(permute(l_pre_map,[2,1,3]));
g_pre_map = caffe('forward_gnet', {gfea1});
g_pre_map = g_pre_map{1};
figure(1011); subplot(1,2,2); imagesc(permute(g_pre_map,[2,1,3]));

l_diff = l_pre_map-permute(single(map1), [2,1,3]);
input_diff = caffe('backward_lnet', {single(l_diff)});
caffe('update_lnet');
g_diff = g_pre_map-permute(single(map1), [2,1,3]);
input_diff = caffe('backward_gnet', {single(g_diff)});
caffe('update_gnet');
fprintf('Iteration %03d/%03d, Local Loss %f, Global Loss %f\n', i, max_iter, sum(abs(l_diff(:))), sum(abs(g_diff(:))));
end
%% ================================================================
t=0;
fnum = size(GT,1);
position = zeros(6, fnum);
update_data = struct('lfea', [], 'map', [], 'conf', [], 'ldistractor', []);

best_geo_param = loc2affgeo(location, pf_param.p_sz);

for im2_id = im1_id:fnum
    l_distractor = false;
    g_distractor = false;
    low_confidence = false;
    caffe('set_phase_test');
    location_last = location;
    tic
    fprintf('Processing Img: %d/%d,       ', im2_id, fnum);
    im2_name = sprintf([data_path 'img/%04d.jpg'], im2_id);
    im2 = double(imread(im2_name));
    if size(im2,3)~=3
        im2(:,:,2) = im2(:,:,1);
        im2(:,:,3) = im2(:,:,1);
    end

    %% extract roi and display
    [roi2, roi_pos, padded_zero_map, pad] = ext_roi(im2, location, l2_off,  roi_size, s2);
    %% draw particles
    geo_param = drawparticals(best_geo_param, pf_param);
    %% preprocess roiroi
    
%     imwrite(mat2gray(roi2), ['material/' num2str(im2_id) '.png']);
    
    roi2 = impreprocess(roi2);   
    fea2 = caffe('forward', {single(roi2)});
    lfea2 = fea2{1};
    lfea2 = lfea2(:,:,lid);
    gfea2 = imresize(fea2{2}, fea_sz(1:2));
    gfea2 = gfea2(:,:,gid);
    %% compute confidence map
    l_pre_map = caffe('forward_lnet', {lfea2});
    l_pre_map = permute(l_pre_map{1}, [2,1,3])/(max(l_pre_map{1}(:))+eps);
    g_pre_map = caffe('forward_gnet', {gfea2});
    g_pre_map = permute(g_pre_map{1}, [2,1,3])/(max(g_pre_map{1}(:))+eps);
    figure(1011); subplot(1,2,1); imagesc(l_pre_map);
    figure(1011); subplot(1,2,2); imagesc(g_pre_map);
    
    if im2_id == 220
        32;
    end
    %% compute global confidence
    g_roi_map = imresize(g_pre_map, roi_pos(4:-1:3));
    g_im_map = padded_zero_map;
    g_im_map(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = g_roi_map;
    g_im_map = g_im_map(pad+1:end-pad, pad+1:end-pad);
    g_im_map = double(g_im_map>0.1).*g_im_map;

    wmaps = warpimg(g_im_map, affparam2mat(geo_param), [pf_param.p_sz, pf_param.p_sz]);
    g_conf = reshape(sum(sum(wmaps))/pf_param.p_sz^2, [], 1);
    g_rank_conf = g_conf.*(pf_param.p_sz^2*geo_param(3,:)'.*geo_param(3,:)'.*geo_param(5,:)').^0.7;
    [~, g_maxid] = max(g_rank_conf);


    
    %% compute local confidence
    l_roi_map = imresize(l_pre_map, roi_pos(4:-1:3));
    l_im_map = padded_zero_map;
    l_im_map(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = l_roi_map;
    l_im_map = l_im_map(pad+1:end-pad, pad+1:end-pad);
    l_im_map = double(l_im_map>0.1).*l_im_map;
    
 
    wmaps = warpimg(l_im_map, affparam2mat(geo_param), [pf_param.p_sz, pf_param.p_sz]);
    l_conf = reshape(sum(sum(wmaps))/pf_param.p_sz^2, [], 1);
    l_rank_conf = l_conf.*(pf_param.p_sz^2*geo_param(3,:)'.*geo_param(3,:)'.*geo_param(5,:)').^0.75;
    [~, l_maxid] = max(l_rank_conf);
    

        %% gnet detect distractor
    potential_location = affgeo2loc(geo_param(:, g_maxid), pf_param.p_sz);
    px1 = min(max(potential_location(1),1), size(im2, 2));
    px2 = min(max(px1+potential_location(3)-1,1), size(im2, 2));
    py1 = min(max(potential_location(2),1), size(im2, 1));
    py2 = min(max(py1+potential_location(4)-1,1), size(im2, 1));
    rectified_im_map = single(g_im_map>0.2).*single(g_im_map);
    inside_conf = sum(sum(rectified_im_map(py1:py2, px1:px2)));
    outside_conf = sum(sum(rectified_im_map)) - inside_conf;
    if outside_conf >= 0.2*inside_conf
        g_distractor = true;
    end
    %% lnet detect distractor
    rectified_im_map = single(l_im_map>0.01).*single(l_im_map);
    inside_conf = sum(sum(rectified_im_map(py1:py2, px1:px2)));
    outside_conf = sum(sum(rectified_im_map)) - inside_conf;
    if outside_conf >= 0.2*inside_conf
        l_distractor = true;
    end
    
    if g_distractor %|| l_distractor
        maxconf = l_conf(l_maxid);
        maxid = l_maxid;
        pre_map = l_roi_map;
    else
        maxconf = g_conf(g_maxid);
        maxid = g_maxid; 
        pre_map = g_roi_map;
    end
    


    fprintf('lmaxconf =  %f,   gmaxconf = %f\n', l_conf(l_maxid),  g_conf(g_maxid));
%     drawresult(im2_id, mat2gray(im2), [pf_param.p_sz, pf_param.p_sz], affparam2mat(geo_param(:, maxid)));
   

    if maxconf>pf_param.mv_thr
    location = affgeo2loc(geo_param(:, maxid), pf_param.p_sz);
    best_geo_param = geo_param(:, maxid);
    elseif l_conf(l_maxid)>pf_param.mv_thr
    location = affgeo2loc(geo_param(:, l_maxid), pf_param.p_sz);
    best_geo_param = geo_param(:, l_maxid);
    maxconf = l_conf(l_maxid);
    end
    
    if maxconf< pf_param.up_thr
        best_geo_param([3,5]) =  position([3,5], im2_id-1);
        location = affgeo2loc( best_geo_param, pf_param.p_sz);
    end
    
    t = t+toc;
    drawresult(im2_id, mat2gray(im2), [pf_param.p_sz, pf_param.p_sz], affparam2mat(best_geo_param));
    position(:, im2_id) = best_geo_param;
    mask = mat2gray(imresize(pre_map, [roi_size, roi_size]));
    pred = grs2rgb(floor(mask*255),jet);
    roi_show = mat2gray(roi2);
 
if im2_id == 267
    267;
end

    
    imwrite([permute(roi_show,[2,1,3]),pred], sprintf('%s/%04d-11.png', res, im2_id));
    figure(112); imshow(mat2gray(rgb2gray(permute(roi_show,[2,1,3])).*double(imresize(pre_map, [roi_size, roi_size])>0.3)));
    tic;
   
        if maxconf>conf_store && maxconf>pf_param.up_thr
            l_off = location_last(1:2)-location(1:2);
            map = GetMap(size(im2), fea_sz, roi_size, location, l_off, s2, 'gaussian'); 
            fea2_store = lfea2;
            map2_store = map;
            conf_store = maxconf; 
            l_distractor_store = l_distractor;
        end

        
        if conf_store>pf_param.up_thr && mod(im2_id,20) == 0 % && ~l_distractor_store
            caffe('set_phase_train');
            caffe('reshape_input', 'lsolver', [0, 2, length(lid), fea_sz(2), fea_sz(1)]);
            fea2_train{1}(:,:,:,1) = lfea1;
            fea2_train{1}(:,:,:,2) = fea2_store;
%             l_pre_map = caffe('forward_lnet', {fea2_store});
            l_pre_map = caffe('forward_lnet', fea2_train);
%             diff = l_pre_map{1}-permute(single(map2_store), [2,1,3]);
            diff{1}(:,:,:,1) = 0.5*(l_pre_map{1}(:,:,:,1)-permute(single(map1), [2,1,3]));
            diff{1}(:,:,:,2) = 0.5*(l_pre_map{1}(:,:,:,2)-permute(single(map2_store), [2,1,3]));
            %         diff = permute((l_pre_map-single(map)).*single(map<=0), [2,1,3]);
            caffe('backward_lnet', diff);
            caffe('update_lnet');
            conf_store = pf_param.up_thr;
            caffe('reshape_input', 'lsolver', [0, 1, length(lid), fea_sz(2), fea_sz(1)]);
        end
        
            if l_distractor && maxconf> pf_param.up_thr
                caffe('set_phase_train');
                caffe('reshape_input', 'lsolver', [0, 2,length(lid), fea_sz(2), fea_sz(1)]);
                %                 lfea2_train(:,:,:,1) = lfea1;
                lfea2_train(:,:,:,1) = fea2_store;
                lfea2_train(:,:,:,2) = lfea2;
                l_off = location_last(1:2)-location(1:2);
                map = GetMap(size(im2), fea_sz, roi_size, location, l_off, s2, 'gaussian');
                iter = 10;
                diff = cell(1);
                for i=1:iter
                    l_pre_map = caffe('forward_lnet', {lfea2_train});
                    diff{1}(:,:,:,1) = 0.5*(l_pre_map{1}(:,:,:,1)-permute(single(map2_store), [2,1,3]));
                    diff{1}(:,:,:,2) = 0.5*squeeze(l_pre_map{1}(:,:,:,2)-permute(single(map),[2,1,3])).*permute(single(map<=0), [2,1,3]);
                    caffe('backward_lnet', diff);
                    caffe('update_lnet');
                end
                caffe('reshape_input', 'lsolver', [0, 1, length(lid), fea_sz(2), fea_sz(1)]);
            end   
    t = t+toc;
    %% save results
    figure(1)
    imwrite(frame2im(getframe(gcf)),sprintf('%s/%04d.png',track_res, im2_id));
    if pf_param.minconf > maxconf
        pf_param.minconf = maxconf;
    end
    if im2_id ==20
        pf_param = reestimate_param(pf_param);
    end
    

end
save([track_res '/position.mat'], 'position');
fprintf('Speed: %d fps\n', fnum/t);
end

function coe = GetCoe(map, fea1, thr)
fg = single(map(:)>0);
bg = single(map(:)<=0);
fm = reshape(fea1, [], size(fea1,3));
coe = single(fm'*(fg-bg)>thr);
figure(2); subplot(1,3,1);imagesc(map);
figure(2); subplot(1,3,2);imagesc(reshape(fm*coe, size(fea1,1), size(fea1,2)));
end

function [ch_w] = GetWeights(map, fea1)
fg = single(map(:)>0);
bg = single(map(:)<=0);
fm = reshape(fea1, [], size(fea1,3));
% ch_w = single(fm'*(fg-bg));
ch_w = single(fm'*(fg));

% figure(2); subplot(1,3,1);imagesc(map);
% figure(2); subplot(1,3,2);imagesc(reshape(fm*coe, size(fea1,1), size(fea1,2)));
end

function select_final = GetSelect1(con, coe, roi, map)
% reduce the top of output layer
select_temp = cell(length(con), 1);
for j = 1 : length(select_temp)
    select_temp{j} = single(ones(length(con(j).contributions), 1));
end
select = select_temp;
select{end+1} = single(coe(:)>0);
select_final{length(select)} = single(coe(:)>0);
caffe('set_select', select);
caffe('reduce');
% compute contribution
fea1 = caffe('forward', {single(roi)});
fea1 = fea1{1};
diff = bsxfun(@times, fea1, permute(single(map), [2,1,3]));
caffe('backward', {diff});
con = caffe('compute_contribution');
% reduce the bottoms layer by layer backward
% portion = [3/3, 1/3, 1/3, 1/2,  1.3/2, 1/2,   1/2,    1.3/2,  1/2,  1/2,  1/3,  1/3,  1/3];
portion = [3/3, 1.3/2, 1.3/2, 1.3/2,  1.3/2, 1.3/2,   1.3/2,    1.3/2,  1.3/2,  1.3/2,  1.3/2,  1.3/2,  1.3/2];
        %  1	1	2 	2	3	3	3	4   4	4	5	    5     5
for i = length(select_temp) : -1 : 1
    % fill in the select 
    select = {};
    for j = 1 : length(select_temp)
        select{j} = single(ones(length(con(j).contributions), 1));
    end
    
    select{end+1} = single(ones(sum(coe), 1));
    % modify the select for current layer
    temp = zeros(length(con(i).contributions), 1);
    [~, id] = sort(con(i).contributions, 'descend');
    temp(id(1:ceil(length(id)*portion(i)))) = 1;
    select{i} = single(temp);
    select_final{i} = single(temp);
    % reduce
    caffe('set_select', select);
    caffe('reduce');
    % compute contribution
    fea1 = caffe('forward', {single(roi(:,:,select{1}>0))});
    fea1 = fea1{1};
    diff = bsxfun(@times, fea1, permute(single(map), [2,1,3]));
    caffe('backward', {diff});
    con = caffe('compute_contribution');
end
% show output
fea1 = permute(fea1, [2,1,3]);
figure(2); subplot(1,3,3);imagesc(sum(fea1, 3));
%% -----
select_final{end+1} = single(ones(128,1));
select_final{end+1} = single(1);
end

function [roi, roi_pos, preim, pad] = ext_roi(im, GT, l_off, roi_size, r_w_scale)
[h, w, ~] = size(im);
win_w = GT(3);
win_h = GT(4);
win_lt_x = GT(1);
win_lt_y = GT(2);
win_cx = round(win_lt_x+win_w/2+l_off(1));
win_cy = round(win_lt_y+win_h/2+l_off(2));
roi_w = r_w_scale(1)*win_w;
roi_h = r_w_scale(2)*win_h;
x1 = win_cx-round(roi_w/2);
y1 = win_cy-round(roi_h/2);
x2 = win_cx+round(roi_w/2);
y2 = win_cy+round(roi_h/2);

im = double(im);
clip = min([x1,y1,h-y2, w-x2]);
pad = 0;
if clip<=0
    pad = abs(clip)+1;
    im = padarray(im, [pad, pad]);
    x1 = x1+pad;
    x2 = x2+pad;
    y1 = y1+pad;
    y2 = y2+pad;
end
roi =  imresize(im(y1:y2, x1:x2, :), [roi_size, roi_size]);
preim = zeros(size(im,1), size(im,2));
roi_pos = [x1, y1, x2-x1+1, y2-y1+1];
% marginl = floor((roi_warp_size-roi_size)/2);
% marginr = roi_warp_size-roi_size-marginl;

% roi = roi(marginl+1:end-marginr, marginl+1:end-marginr, :);
% roi = imresize(roi, [roi_size, roi_size]);
end


function I = impreprocess(im)
mean_pix = [103.939, 116.779, 123.68]; % BGR
im = permute(im, [2,1,3]);
im = im(:,:,3:-1:1);
I(:,:,1) = im(:,:,1)-mean_pix(1); % substract mean
I(:,:,2) = im(:,:,2)-mean_pix(2);
I(:,:,3) = im(:,:,3)-mean_pix(3);
end

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

function I = crop_bg(im, GT, mean_pix)
[im_h, im_w, ~] = size(im);
win_w = GT(3);
win_h = GT(4);
win_lt_x = max(GT(1), 1);
win_lt_x = min(im_w, win_lt_x);
win_lt_y = max(GT(2), 1);
win_lt_y = min(im_h, win_lt_y);

win_rb_x = max(win_lt_x+win_w-1, 1);
win_rb_x = min(im_w, win_rb_x);
win_rb_y = max(win_lt_y+win_h-1, 1);
win_rb_y = min(im_h, win_rb_y);

I = zeros(im_h, im_w, 3);
I(:,:,1) = mean_pix(3);
I(:,:,2) = mean_pix(2);
I(:,:,3) = mean_pix(1);
I(win_lt_y:win_rb_y, win_lt_x:win_rb_x, :) = im(win_lt_y:win_rb_y, win_lt_x:win_rb_x, :);
end

function param = loc2affgeo(location, p_sz)
% location = [tlx, tly, w, h]

cx = location(1)+(location(3)-1)/2;
cy = location(2)+(location(4)-1)/2;
param = [cx, cy, location(3)/p_sz, 0, location(4)/location(3), 0]';

end


function   location = affgeo2loc(geo_param, p_sz)
w = geo_param(3)*p_sz;
h = w*geo_param(5);
tlx = geo_param(1) - (w-1)/2;
tly = geo_param(2) - (h-1)/2;
location = round([tlx, tly, w, h]);
end


function geo_params = drawparticals(geo_param, pf_param)
geo_param = repmat(geo_param, [1,pf_param.p_num]);
geo_params = geo_param + randn(6,pf_param.p_num).*repmat(pf_param.affsig(:),[1,pf_param.p_num]);
end


function drawresult(fno, frame, sz, mat_param)
figure(1); clf;
set(gcf,'DoubleBuffer','on','MenuBar','none');
colormap('gray');
axes('position', [0 0 1 1])
imagesc(frame, [0,1]); hold on;
text(5, 18, num2str(fno), 'Color','y', 'FontWeight','bold', 'FontSize',18);
drawbox(sz(1:2), mat_param, 'Color','r', 'LineWidth',2.5);
axis off; hold off;
drawnow;
end

function [sal, id] = compute_saliency(fea1, map, solver)
caffe('set_phase_test');
if strcmp(solver, 'lsolver')
    out = caffe('forward_lnet', fea1);
    diff1 = {out{1}-permute(single(map), [2,1,3])};
    input_diff1 = caffe('backward_lnet', diff1);
    diff2 = {single(ones(size(fea1{1},1)))};
    input_diff2 = caffe('backward2_lnet', diff2);
elseif strcmp(solver, 'gsolver')
    out = caffe('forward_gnet', fea1);
    diff2 = {single(ones(size(fea1{1},1)))};
    diff1 = {out{1}-permute(single(map), [2,1,3])};
    input_diff1 = caffe('backward_gnet', diff1);
    input_diff2 = caffe('backward2_gnet', diff2);
else
    error('Unkonwn solver type')
end
% sal = sum(sum(input_diff2{1}.*(fea1{1}).^2));
% sal = -sum(sum(input_diff1{1}.*fea1{1}))+0.5*sum(sum(input_diff2{1}.*(fea1{1}).^2));
sal = -sum(sum(input_diff1{1}.*fea1{1}+0.5*input_diff2{1}.*(fea1{1}).^2));

sal = sal(:);
[~, id] = sort(sal, 'descend');
end
