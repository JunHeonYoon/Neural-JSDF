ax_handle = axes('View',[115 12],'Position',[0.1300 0.1100 0.7750 0.8150]);
axis off
hold on 
axis equal
load('meshes/mesh_full.mat');
addpath('functions/')


base = eye(4);
joint_state = [0 0 0 -pi/2 0 pi/2 0 0];
% clr = [0 0.5 1];
clr = [0 0.4470 0.7410;
       0.8500 0.3250 0.0980;
       0.4940 0.1840 0.5560;
       0.9290 0.6940 0.1250;
       0.6350 0.0780 0.1840;
       0.3010 0.7450 0.9330;
       0.4660 0.6740 0.1880;
       0.47 0.25 0.80;
       0.9856 0.5372 0.0537];
facealpha_ = 1;
patch_handle = [];
hl_idx = [];
%
r = [0 0 0 0.0825 -0.0825 0 0.088 0];
d = [0.333 0 0.316 0 0.384 0 0 0.107];
alpha = [0 -pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2 0];

%r = [0 0 0 0.0825 -0.0825 0 0.088];
%d = [0.333 0 0.316 0 0.384 0 0];
%alpha = [0 -pi/2 pi/2 pi/2 -pi/2 pi/2 pi/2];

P = franka_dh_fk(joint_state, r, d, alpha,base);
init = 0;
if isempty(patch_handle)
    init = 1;
end
%hl = {};
%hl = {'hand';'link3'};
%% plotting
%uiaxes(ax_handle);
facecolor = clr;
facealpha = facealpha_;

edgealpha = 0;
idx_order ={'link0';'link1';'link2';'link3';'link4';'link5';'link6';'link7';'hand';'finger';'finger'};
idx_order ={'link0';'link1';'link2';'link3';'link4';'link5';'link6';'link7';'hand'};

%plot base to end-effector (without fingers!)
for i = 1:1:length(idx_order)
    idx = find_idx(idx_order{i},mesh);
    R = P{i}(1:3,1:3);
    T = P{i}(1:3,4);
    V = mesh{idx}.v*R'+T';
    %V = (P{i}*[mesh{idx}.v ones(size(mesh{idx}.v,1),1)]')';
    F = mesh{idx}.f;
%     hl_idx
    if ismember(idx_order{i},idx_order(hl_idx))
        facecolor = [1 0 0];
    else
        facecolor = clr(i, :);
    end
    if init
        patch_handle{i} = patch(ax_handle,'Faces',F,'Vertices',V(:,1:3),'facecolor',facecolor,'Facealpha',facealpha,'edgealpha',edgealpha);
    else
        patch_handle{i}.Vertices = V(:,1:3);
        patch_handle{i}.FaceColor = facecolor;
    end
end
camlight