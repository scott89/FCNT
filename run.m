addpath('caffe-fcnt/matlab/caffe/', 'util');
data_path = ['video/'];
seq_name = 'MotorRolling';
seq.path = ['video/' seq_name '/'];

if ~isdir(seq.path)
    system('sh ./video/download_motorrolling.sh');
end
gt = load([seq.path 'groundtruth_rect.txt']);
seq.init_rect = gt(1,:);
seq.startFrame = 1;
seq.endFrame = size(gt, 1);
track_param = init_tracker(seq);
position = cnn2_pf_tracker(track_param);
caffe('reset');

