function  position = cnn2_pf_tracker(tracker_param)

load_tracker_param;
caffe('presolve_gnet');
caffe('presolve_snet');
%% read images
im1_name = sprintf([data_path 'img/%04d.jpg'], im1_id);
im1 = double(imread(im1_name));
if size(im1,3)~=3
    im1(:,:,2) = im1(:,:,1);
    im1(:,:,3) = im1(:,:,1);
end

%% extract roi and display
roi1 = ext_roi(im1, location, [0, 0],  roi_size, s1);
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
map1 =  GetMap(size(im1), fea_sz, roi_size, location, [0, 0], s1, 'gaussian');
conf_store = 0;

%% select
caffe('set_phase_train');
for i=1:max_iter_select
s_pre_map = caffe('forward_snet', {lfea1});
s_pre_map = s_pre_map{1};
figure(1011); subplot(1,2,1); imagesc(permute(s_pre_map,[2,1,3]));
s_diff = s_pre_map-permute(single(map1), [2,1,3]);
caffe('backward_snet', {single(s_diff)});
caffe('update_snet');

g_pre_map = caffe('forward_gnet', {gfea1});
g_pre_map = g_pre_map{1};
figure(1011); subplot(1,2,2); imagesc(permute(g_pre_map,[2,1,3]));
g_diff = g_pre_map-permute(single(map1), [2,1,3]);
caffe('backward_gnet', {single(g_diff)});
caffe('update_gnet');
fprintf('Iteration %03d/%03d, Local Loss %f, Global Loss %f\n', i, max_iter, sum(abs(s_diff(:))), sum(abs(g_diff(:))));
end


[~, lid] = compute_saliency({lfea1}, map1, 'ssolver');
[~, gid] = compute_saliency({gfea1}, map1, 'gsolver');

lid = lid(1:ch_num);
gid = gid(1:ch_num);

lfea1 = lfea1(:,:,lid);
gfea1 = gfea1(:,:,gid);
fea2_store = lfea1;
map2_store = map1;
%% train
caffe('init_ssolver', snet_solver_def_file);
caffe('init_gsolver', gnet_solver_def_file);
caffe('set_phase_train');
caffe('presolve_gnet');
caffe('presolve_snet');
for i=1:max_iter
s_pre_map = caffe('forward_snet', {lfea1});
s_pre_map = s_pre_map{1};
figure(1011); subplot(1,2,1); imagesc(permute(s_pre_map,[2,1,3]));
g_pre_map = caffe('forward_gnet', {gfea1});
g_pre_map = g_pre_map{1};
figure(1011); subplot(1,2,2); imagesc(permute(g_pre_map,[2,1,3]));

s_diff = s_pre_map-permute(single(map1), [2,1,3]);
input_diff = caffe('backward_snet', {single(s_diff)});
caffe('update_snet');
g_diff = g_pre_map-permute(single(map1), [2,1,3]);
input_diff = caffe('backward_gnet', {single(g_diff)});
caffe('update_gnet');
fprintf('Iteration %03d/%03d, Local Loss %f, Global Loss %f\n', i, max_iter, sum(abs(s_diff(:))), sum(abs(g_diff(:))));
end
%% ================================================================
t=0;

position = zeros(6, fnum);
best_geo_param = loc2affgeo(location, pf_param.p_sz);
for im2_id = im1_id:fnum
    s_distractor = false;
    g_distractor = false;
    caffe('set_phase_test');
    location_last = location;
    fprintf('Processing Img: %d/%d,       ', im2_id, fnum);
    im2_name = sprintf([data_path 'img/%04d.jpg'], im2_id);
    im2 = double(imread(im2_name));
    if size(im2,3)~=3
        im2(:,:,2) = im2(:,:,1);
        im2(:,:,3) = im2(:,:,1);
    end

    %% extract roi and display
    [roi2, roi_pos, padded_zero_map, pad] = ext_roi(im2, location, [0, 0],  roi_size, s2);
    %% draw particles
    geo_param = drawparticals(best_geo_param, pf_param);
    %% preprocess roiroi
    
    roi2 = impreprocess(roi2);   
    fea2 = caffe('forward', {single(roi2)});
    lfea2 = fea2{1};
    lfea2 = lfea2(:,:,lid);
    gfea2 = imresize(fea2{2}, fea_sz(1:2));
    gfea2 = gfea2(:,:,gid);
    %% compute confidence map
    s_pre_map = caffe('forward_snet', {lfea2});
    s_pre_map = permute(s_pre_map{1}, [2,1,3])/(max(s_pre_map{1}(:))+eps);
    g_pre_map = caffe('forward_gnet', {gfea2});
    g_pre_map = permute(g_pre_map{1}, [2,1,3])/(max(g_pre_map{1}(:))+eps);
    figure(1011); subplot(1,2,1); imagesc(s_pre_map);
    figure(1011); subplot(1,2,2); imagesc(g_pre_map);

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
    s_roi_map = imresize(s_pre_map, roi_pos(4:-1:3));
    s_im_map = padded_zero_map;
    s_im_map(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = s_roi_map;
    s_im_map = s_im_map(pad+1:end-pad, pad+1:end-pad);
    s_im_map = double(s_im_map>0.1).*s_im_map;
    
 
    wmaps = warpimg(s_im_map, affparam2mat(geo_param), [pf_param.p_sz, pf_param.p_sz]);
    s_conf = reshape(sum(sum(wmaps))/pf_param.p_sz^2, [], 1);
    s_rank_conf = s_conf.*(pf_param.p_sz^2*geo_param(3,:)'.*geo_param(3,:)'.*geo_param(5,:)').^0.75;
    [~, s_maxid] = max(s_rank_conf);
    

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
    %% snet detect distractor
    rectified_im_map = single(s_im_map>0.01).*single(s_im_map);
    inside_conf = sum(sum(rectified_im_map(py1:py2, px1:px2)));
    outside_conf = sum(sum(rectified_im_map)) - inside_conf;
    if outside_conf >= 0.2*inside_conf
        s_distractor = true;
    end
    
    if g_distractor %|| s_distractor
        maxconf = s_conf(s_maxid);
        maxid = s_maxid;
        pre_map = s_roi_map;
    else
        maxconf = g_conf(g_maxid);
        maxid = g_maxid; 
        pre_map = g_roi_map;
    end

    fprintf('lmaxconf =  %f,   gmaxconf = %f\n', s_conf(s_maxid),  g_conf(g_maxid));
   
    if maxconf>pf_param.mv_thr
    location = affgeo2loc(geo_param(:, maxid), pf_param.p_sz);
    best_geo_param = geo_param(:, maxid);
    elseif s_conf(s_maxid)>pf_param.mv_thr
    location = affgeo2loc(geo_param(:, s_maxid), pf_param.p_sz);
    best_geo_param = geo_param(:, s_maxid);
    maxconf = s_conf(s_maxid);
    end
    
    if maxconf< pf_param.up_thr
        best_geo_param([3,5]) =  position([3,5], im2_id-1);
        location = affgeo2loc( best_geo_param, pf_param.p_sz);
    end
    

    drawresult(im2_id, mat2gray(im2), [pf_param.p_sz, pf_param.p_sz], affparam2mat(best_geo_param));
    position(:, im2_id) = best_geo_param;
    mask = mat2gray(imresize(pre_map, [roi_size, roi_size]));
    pred = grs2rgb(floor(mask*255),jet);
    roi_show = mat2gray(roi2);

    figure(112); imshow(mat2gray(rgb2gray(permute(roi_show,[2,1,3])).*double(imresize(pre_map, [roi_size, roi_size])>0.3)));

   
        if maxconf>conf_store && maxconf>pf_param.up_thr
            l_off = location_last(1:2)-location(1:2);
            map = GetMap(size(im2), fea_sz, roi_size, location, l_off, s2, 'gaussian'); 
            fea2_store = lfea2;
            map2_store = map;
            conf_store = maxconf; 
        end

        
        if conf_store>pf_param.up_thr && mod(im2_id,20) == 0 % && ~l_distractor_store
            caffe('set_phase_train');
            caffe('reshape_input', 'ssolver', [0, 2, length(lid), fea_sz(2), fea_sz(1)]);
            fea2_train{1}(:,:,:,1) = lfea1;
            fea2_train{1}(:,:,:,2) = fea2_store;

            s_pre_map = caffe('forward_snet', fea2_train);
            diff{1}(:,:,:,1) = 0.5*(s_pre_map{1}(:,:,:,1)-permute(single(map1), [2,1,3]));
            diff{1}(:,:,:,2) = 0.5*(s_pre_map{1}(:,:,:,2)-permute(single(map2_store), [2,1,3]));
            caffe('backward_snet', diff);
            caffe('update_snet');
            conf_store = pf_param.up_thr;
            caffe('reshape_input', 'ssolver', [0, 1, length(lid), fea_sz(2), fea_sz(1)]);
        end
        
            if s_distractor && maxconf> pf_param.up_thr
                caffe('set_phase_train');
                caffe('reshape_input', 'ssolver', [0, 2,length(lid), fea_sz(2), fea_sz(1)]);
                lfea2_train(:,:,:,1) = fea2_store;
                lfea2_train(:,:,:,2) = lfea2;
                l_off = location_last(1:2)-location(1:2);
                map = GetMap(size(im2), fea_sz, roi_size, location, l_off, s2, 'gaussian');
                iter = 10;
%                 diff = cell(1);
                for i=1:iter
                    s_pre_map = caffe('forward_snet', {lfea2_train});
                    diff{1}(:,:,:,1) = 0.5*(s_pre_map{1}(:,:,:,1)-permute(single(map2_store), [2,1,3]));
                    diff{1}(:,:,:,2) = 0.5*squeeze(s_pre_map{1}(:,:,:,2)-permute(single(map),[2,1,3])).*permute(single(map<=0), [2,1,3]);
                    caffe('backward_snet', diff);
                    caffe('update_snet');
                end
                caffe('reshape_input', 'ssolver', [0, 1, length(lid), fea_sz(2), fea_sz(1)]);
            end   
    %% save results

    if pf_param.minconf > maxconf
        pf_param.minconf = maxconf;
    end
    if im2_id ==20
        pf_param = reestimate_param(pf_param);
    end
    
end
% save([track_res '/position.mat'], 'position');
end















