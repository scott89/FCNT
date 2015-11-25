pf_param = struct('affsig', [10,10,.004,.00,0.00,0], 'p_sz', 64,...
            'p_num', 600, 'mv_thr', 0.1, 'up_thr', 0.35, 'pos_thr', 0.3, 'roi_scale', 2);

res = ['select_roi_cn' num2str(ch_num) '/' set_name];


data_path = ['video/' set_name '/'];
GT = load([data_path 'groundtruth_rect.txt']);
dia = (GT(1, 3)^2+GT(1, 4)^2)^0.5;

% scale = gt(1, 3)/ gt (1, 4);
scale = [dia/GT(1, 3), dia/GT(1, 4)];
l1_off = [0,0];
l2_off = [0,0];
s1 = pf_param.roi_scale*[scale(1),scale(2)];
s2 = pf_param.roi_scale*[scale(1),scale(2)];

feature_solver_def_file = 'feature_solver.prototxt';

gnet_solver_def_file = ['solver/gnet_solver_' num2str(ch_num) '.prototxt'];
lnet_solver_def_file = ['solver/lnet_solver_' num2str(ch_num) '.prototxt']; 
select_lnet_solver_def_file = 'select_lnet_solver.prototxt'; 
select_gnet_solver_def_file = 'select_gnet_solver.prototxt';
model_file = 'VGG_ILSVRC_16_layers.caffemodel';
caffe('init_solver', feature_solver_def_file, model_file);
caffe('init_gsolver', select_gnet_solver_def_file, model_file);
caffe('init_lsolver', select_lnet_solver_def_file, model_file);
caffe('set_mode_gpu');

roi_size = 368;

mean_pix = [103.939, 116.779, 123.68]; 

location = GT(im1_id,:);

pf_param.ratio = location(3)/pf_param.p_sz;
pf_param.affsig(3) = pf_param.affsig(3)*pf_param.ratio;
pf_param.affsig_o = pf_param.affsig;
pf_param.affsig(3) = 0;
pf_param.minconf = 0.5;
