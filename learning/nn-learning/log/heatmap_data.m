clear all;close all;clc

TNR = [[0.869636,0.877565,0.790578,0.958722,0.839086,0.950326,0.958256,0.945662],
       [0.501975,0.819996,0.883600,0.917065,0.890667,0.952401,0.957181,0.957597],
       [0.538187,0.894862,0.874430,0.813926,0.916286,0.946439,0.951002,0.965285],
       [0.468683,0.656587,0.834086,0.827999,0.877086,0.964461,0.951109,0.911447],
       [0.544868,0.556403,0.836950,0.831470,0.816031,0.946432,0.955621,0.924927],
       [0.198043,0.276868,0.504448,0.734698,0.725801,0.859964,0.881495,0.866014],
       [0.082100,0.273721,0.416050,0.791218,0.853297,0.860700,0.884758,0.857335],
       [0.031357,0.335332,0.473783,0.627485,0.777416,0.832077,0.868403,0.816484],
       [0.249829,0.418095,0.478753,0.519020,0.706648,0.817341,0.851953,0.748286]];
TPR = [[0.990976,0.993962,0.998186,0.995665,0.999359,0.998828,0.999160,0.999182],
       [0.989953,0.989975,0.994227,0.994204,0.998546,0.998165,0.998411,0.997270],
       [0.992105,0.980566,0.992577,0.996963,0.997256,0.997728,0.998043,0.995771],
       [0.983674,0.983606,0.987412,0.993154,0.995947,0.994866,0.996239,0.996780],
       [0.979047,0.985017,0.980556,0.988960,0.997341,0.994030,0.995179,0.995562],
       [0.991135,0.994303,0.982384,0.973131,0.993870,0.990611,0.991249,0.990998],
       [0.992010,0.985972,0.984572,0.956194,0.977684,0.986431,0.988475,0.985857],
       [0.998030,0.974693,0.980304,0.976457,0.985549,0.991847,0.992236,0.989786],
       [0.982571,0.969678,0.980258,0.982984,0.987702,0.986350,0.988068,0.987266]];

xvalue = ["256", "512", "1024", "2048", "4096", "1e5", "5e5", "1e6"];
yvalue = [1,2,3,4,5,6,7,8,9];



% figure("Name","Heatmap of TPR/TNR")
% subplot(1,2,1)
% h1 = heatmap(xvalue, yvalue, TPR);
% h1.ColorScaling = "scaledrows";
% h1.Title = "TPR"; h1.XLabel = "Batch size"; h1.YLabel = "Link index";
% h1.FontSize = 15;
% % h1.ColorbarVisible = 'off';
% subplot(1,2,2)
% h2 = heatmap(xvalue, yvalue, TNR);
% h2.ColorScaling = "scaledrows";
% h2.Title = "TNR"; h2.XLabel = "Batch size"; h2.YLabel = "Link index";
% h2.FontSize = 15;
% h2.Colormap = flipud(autumn);
% % h2.ColorbarVisible = 'off';


figure()
b1 = bar3(TNR);
ax1 = gca;
ax1.YLabel.String = 'Link index';
ax1.YLabel.FontSize = 25;
ax1.XLabel.String = 'Batch Size';
ax1.XLabel.FontSize = 25;
ax1.ZLabel.String = 'TNR';
ax1.ZLabel.FontSize = 25;
ax1.FontWeight = "bold";
ax1.XTickLabel = xvalue;
ax1.YTickLabel = yvalue;
ax1.FontName = 'Times';
saveas(gca,"figure/bar plot(TNR).svg")

figure()
b2 = bar3(TPR);
ax2 = gca;
ax2.YLabel.String = 'Link index';
ax2.YLabel.FontSize = 25;
ax2.XLabel.String = 'Batch Size';
ax2.XLabel.FontSize = 25;
ax2.ZLabel.String = 'TPR';
ax2.ZLabel.FontSize = 25;
ax2.ZLim = [0.95 1];
ax2.FontWeight = "bold";
ax2.XTickLabel = xvalue;
ax2.YTickLabel = yvalue;
saveas(gca,"figure/bar plot(TPR).svg")