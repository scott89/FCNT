function [ res] = run_FCNT(seq, res_path, bSaveImage )
close all
cd tracker/FCNT/
addpath('caffe-fcnt/matlab/caffe/','util/');
tracker_param = init_tracker(seq);
tracker_param.startFrame = seq.startFrame;
tracker_param.endFrame = seq.endFrame;
% tracker_param.endFrame = 20;
tracker_param.init_rect = seq.init_rect;
positions = cnn2_pf_tracker(tracker_param);
res.type = 'ivtAff';
res.res = affparam2mat(positions);
res.res = res.res';
res.tmplsize = [64,64];
cd ../../;
end

