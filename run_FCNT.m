function [ res] = run_FCNT(seq, res_path, bSaveImage )
close all
cd tracker/FCNT/
addpath('caffe-fcnt/matlab/caffe/','util/');
tracker_param = init_tracker(seq);
positions = cnn2_pf_tracker(tracker_param);
res.type = 'ivtAff';
res.res = affparam2mat(positions);
res.res = res.res';
res.tmplsize = [64,64];
cd ../../;
end

