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