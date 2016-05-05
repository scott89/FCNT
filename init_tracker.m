function tracker_param = init_tracker(seq)
%% particle filter parameters
pf_param = struct('affsig', [10,10,.004,.00,0.00,0], 'p_sz', 64,...
            'p_num', 700, 'mv_thr', 0.1, 'up_thr', 0.35, 'roi_scale', 2);
%% check if sequence exists
seq_path = seq.path;
tracker_param.seq_path = seq_path;

%% parameters to crop ROI
location = seq.init_rect;
pf_param.affsig(1) = ceil((location(3)^2 + location(4)^2)^0.5/7);
% pf_param.affsig(1) = max(min(location(3), location(4))/4, 7);
pf_param.affsig(2) = pf_param.affsig(1);
% pf_param.p_num = floor(7 * pf_param.affsig(1) * pf_param.affsig(2));
tracker_param.location = location;
dia = (location(3)^2 + location(4)^2)^0.5;
scale = [dia / location(3), dia / location(4)];
% tracker_param.l1_off = [0,0];
% tracker_param.l2_off = [0,0];
tracker_param.s1 = pf_param.roi_scale*[scale(1),scale(2)];
tracker_param.s2 = pf_param.roi_scale*[scale(1),scale(2)];
tracker_param.roi_size = 368;
pf_param.ratio = location(3)/pf_param.p_sz;
pf_param.affsig(3) = pf_param.affsig(3)*pf_param.ratio;
pf_param.affsig_o = pf_param.affsig;
pf_param.affsig(3) = 0;
pf_param.minconf = 0.5;
tracker_param.pf_param = pf_param;
%% init feature net and sel-cnn 
tracker_param.ch_num = 384; %% number of selected channels;

feature_solver_def_file = './model/feature_solver.prototxt';
model_file = '/home/lijun/Research/Code/FCT_scale_base/model/VGG_ILSVRC_16_layers.caffemodel';
caffe('init_solver', feature_solver_def_file, model_file);

select_snet_solver_def_file = 'solver/select_snet_solver.prototxt'; 
select_gnet_solver_def_file = 'solver/select_gnet_solver.prototxt';
caffe('init_gsolver', select_gnet_solver_def_file);
caffe('init_ssolver', select_snet_solver_def_file);
caffe('set_mode_gpu');

%% gnet and snet solver file
tracker_param.gnet_solver_def_file = ['solver/gnet_solver_' num2str(tracker_param.ch_num) '.prototxt'];
tracker_param.snet_solver_def_file = ['solver/snet_solver_' num2str(tracker_param.ch_num) '.prototxt']; 

% %% results path 
% tracker_param.result_path = 'results/'
% if ~isdir(tracker_param.result_path)
%     mkdir(tracker_param.result_path);
% end




    

