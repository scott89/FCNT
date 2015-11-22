clc
% path = 'cnn_res/';
num = 384;
path = ['select_cnn_res_cn' num2str(num) '/'];

dataSet = dir(path);


% dataSet.name = 'Basketball';
% resPath = ['select_cnn' num2str(num) '_TBres/'];
resPath = ['fcn7_restore_' num2str(num) '_TBres/'];

if ~isdir(resPath)
    mkdir(resPath);
end

for idSeq = 1:length(dataSet)
    if ~isdir([path dataSet(idSeq).name]) || strcmp(dataSet(idSeq).name,'.') || strcmp(dataSet(idSeq).name,'..')
        continue;
    end
    load([path dataSet(idSeq).name '/position.mat']);
    results=cell(1);
    results{1}.res = position(:,1:end)';
    results{1}.type = 'ivtAff';
    results{1}.tmplsize = [64,64];
    results{1}.startFame = 1;
    results{1}.annoBegin = 1;
    results{1}.len = length(results{1}.res);
    results{1}.property = 'geom';
    
        
    if strcmp(dataSet(idSeq).name, 'Tiger1')
        results{1}.startFame = 6;
        results{1}.res = results{1}.res(6:end,:);
        results{1}.len = length(results{1}.res);
    end
%     save([resPath lower(dataSet(idSeq).name) '_select_cnn' num2str(num) '-7.mat'], 'results')
    save([resPath lower(dataSet(idSeq).name) '_fcn7_restore_' num2str(num) '.mat'], 'results')

end