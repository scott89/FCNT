function drawopt = drawtrackresult(drawopt, fno, frame, tmpl, param)
% function drawopt = drawtrackresult(drawopt, fno, frame, tmpl, param, pts)
%
%   drawopt : misc info for drawing, intitially []
%         [.showcoef] : shows coefficient
%         [.showcondens,thcondens] : show condensation candidates
%   fno : frame number
%   frame(fh,fw) : current frame
%   tmpl.mean(th,tw) : mean image
%       .basis(tN,nb) : basis
%   param.est : current estimate
%        .wimg : warped image
%       [.err,mask] : error, mask image
%       [.param,conf] : condensation
%
% uses: util/showimgs

% Copyright (C) 2005 Jongwoo Lim and David Ross.
% All rights reserved.



  figure(1); clf;
  set(gcf,'DoubleBuffer','on','MenuBar','none');
  colormap('gray');


sz = size(tmpl.mean);  

axes('position', [0 0 1 1])
imagesc(frame, [0,1]); hold on;

text(5, 18, num2str(fno), 'Color','y', 'FontWeight','bold', 'FontSize',18);
drawbox(sz(1:2), param.est, 'Color','r', 'LineWidth',2.5);
axis off; hold off;

drawnow;
