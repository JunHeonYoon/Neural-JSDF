clc
clear all
close all
%%
%% visualization of sampled data. Specify any index below:
idx_plot = 1;
%%
b=4096;
load(strcat('datasets/data_mesh(size=', int2str(b),').mat'));
jpos = dataset(:,1:10);
dists = dataset(:,11:end);
n_links = size(dists,2);

ax_body = axes('View',[115 12],'Position',[0.1300 0.1100 0.7750 0.8150]);
axis off
hold on 
axis equal
load('meshes/mesh_light.mat');

mesh_fk = meshes_fk(mesh, eye(4), [jpos(1,1:7) 0]);
N_MESHES = length(mesh_fk)-2;

tmp_handle = plot_franka_fcn(ax_body,[], mesh, eye(4), [jpos(1,1:7) 0],[0 0.5 1],[], 0);
closepts_arr = [];
for i = 1:length(dataset)
    hold on
    if min(abs(dists(i, :))) < 0.001
        closepts_arr = [closepts_arr ; jpos(i, 8:10)];
        plot3(jpos(i,8),jpos(i,9),jpos(i,10),'r.')
    end
end
% shp = alphaShape(closepts_arr(:,1), closepts_arr(:,2), closepts_arr(:,3), 0.04);
% [tri, xyz] = boundaryFacets(shp);
% plot(shp, "LineStyle","none", "FaceColor","r", "FaceAlpha",0.4)
camlight
exportgraphics(gcf,strcat('figure/batch(',int2str(b),')_wo.png'),'Resolution',600)