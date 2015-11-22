function deim = deconv(i,l1, l2, roi1)
fea1 = caffe('forward', {single(roi1)});
fm = fea1{l1}(:,:,i);
figure(1); imagesc(permute(fm, [2,1,3]))
% [maxact, maxid] = max(fm(:));
% fm = zeros(size(fm));
% [r,c] = ind2sub(size(fm), maxid);
% fm(r,c) = maxact;
fea1{l1} = single(zeros(size(fea1{l1})));
fea1{l1}(:,:,i)=fm;
fea1{l2} = single(zeros(size(fea1{l2})));


deim = caffe('backward', fea1);
deim = permute(deim{1}, [2,1,3]);
deim = deim(:,:,3:-1:1);
roi1 = permute(roi1, [2,1,3]);
roi1 = roi1(:,:,3:-1:1);
% figure(2);imshow(permute(mat2gray(mat2gray(roi1).*deim),[2,1,3]))

% deim = deim.*double(deim>mean(deim(:)));
% deim = deim.*double(deim>0);


figure(2);imshow(mat2gray(deim))
for ch = 1:3
figure(3);subplot(2,2,ch); imshow(mat2gray(deim(:,:,ch)));
end
figure(3);subplot(2,2,4); imshow(mat2gray(roi1));
% figure(4);imshow(mat2gray(mat2gray(permute(roi1(:,:,1),[2,1,3])).*deim(:,:,1)));