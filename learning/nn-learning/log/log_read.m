clc; clear all; close all

batch_size_list = [256, 512, 1024, 2048, 4096, 10000, 50000, 100000];
% batch_size_list = [256, 512, 1024, 2048, 4096, 10000, 50000, 100000, 150000,200000,250000,300000];
% dropout_list = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1];

for i=1:length(batch_size_list)
    filename(i, 1) = strcat("log(b=", num2str(batch_size_list(i)), ").txt");
    fileID(i) = fopen(filename(i, 1));
    data= textscan(fileID(i),'Epoch: %f (Saved at %f), Train Loss: %f, Validation Loss: %f (%f), Epoch time: %f s, LR = %f');
    for j=1:7
        log{i}(:, j) = data{j};
    end
    fclose(fileID(i));
end
% for i=1:length(dropout_list)
%     filename(i, 1) = strcat("log(b=50000)_dropout_", num2str(dropout_list(i)), ".txt");
%     fileID(i) = fopen(filename(i, 1));
%     data= textscan(fileID(i),'Epoch: %f (Saved at %f), Train Loss: %f, Validation Loss: %f (%f), Epoch time: %f s, LR = %f');
%     for j=1:7
%         log{i}(:, j) = data{j};
%     end
%     fclose(fileID(i));
% end


% log cell
% % epoch, saved epoch, train loss, validation loss, 
% % end effector close to pts loss, epoch time, LR
for i=1:length(log)
    f = figure("Name",strcat("Loss (batch size = " + num2str(batch_size_list(i)) + ")"));
    f.Position(3:4) = [600,300];
    % figure("Name",strcat("Loss (dropout = " + num2str(dropout_list(i)) + ")"))
    plot(log{i}(:,1), log{i}(:,3), "r-"); hold on
    plot(log{i}(:,1), log{i}(:,4), "b-"); hold off
    title("Loss (batch size = " + num2str(batch_size_list(i)) + ")", "FontWeight","bold","FontName", 'Times')
    % title("Loss (dropout = " + num2str(dropout_list(i)) + ")")
    xlabel("Epoch", "FontWeight","bold"); ylabel("Loss[cm^2]", "FontWeight","bold"); xlim tight;ylim([0 80]);
    legend(["Train", "Validation"], "FontWeight","bold")
    % if i>=6
    %     ylim([0,30]);
    % end
    set(gca,"FontName", 'Times')
    saveas(gcf, strcat("figure/Loss(batch size = " + num2str(batch_size_list(i)) + ").svg"));
    % saveas(gcf, strcat("figure/Loss(dropout = " + num2str(dropout_list(i)) + ").svg"));

    disp("Training Time[h] : ")
    disp(sum(log{i}(:,6))/3600)
end

% f=figure("Name", "Loss")
% f.Position(3:4) = [600,300];
% subplot(2,2,1)
% plot(log{1}(:,1), log{1}(:,3), "r-"); hold on
% plot(log{1}(:,1), log{1}(:,4), "b-"); hold off
% title("b = 1024"); grid on
% xlabel("Epoch", "FontWeight","bold"); ylabel("Loss[cm^2]", "FontWeight","bold"); xlim tight;ylim([0 80]);legend(["Train", "Validation"])
% subplot(2,2,2)
% plot(log{5}(:,1), log{5}(:,3), "r-"); hold on
% plot(log{5}(:,1), log{5}(:,4), "b-"); hold off
% title("b = 4096"); grid on
% xlabel("Epoch", "FontWeight","bold"); ylabel("Loss[cm^2]", "FontWeight","bold"); xlim tight;ylim([0 80]);legend(["Train", "Validation"])
% subplot(2,2,3)
% plot(log{7}(:,1), log{7}(:,3), "r-"); hold on
% plot(log{7}(:,1), log{7}(:,4), "b-"); hold off
% title("b = 50000"); grid on
% xlabel("Epoch", "FontWeight","bold"); ylabel("Loss[cm^2]", "FontWeight","bold"); xlim tight;ylim([0 80]);legend(["Train", "Validation"])
% subplot(2,2,4)
% plot(log{8}(:,1), log{8}(:,3), "r-"); hold on
% plot(log{8}(:,1), log{8}(:,4), "b-"); hold off
% title("b = 100000"); grid on
% xlabel("Epoch", "FontWeight","bold"); ylabel("Loss[cm^2]", "FontWeight","bold"); xlim tight;ylim([0 80]);legend(["Train", "Validation"])

% test_loss = [];
% val_loss = [];
% for i=1:length(log)
%     test_loss(i,1) = log{i}(log{i}(end,2),3);
%     val_loss(i,1) = log{i}(log{i}(end,2),4);
% end
% semilogy(1:length(log), test_loss, "LineWidth",2); hold on
% semilogy(1:length(log), val_loss, "LineWidth",2); hold off
% set(gca, 'XTick', 1:length(log) , 'XTickLabel', batch_size_list)
% title("Loss per Batch size", "FontSize", 20)
% xlim tight; xlabel("Batch size", "FontWeight","bold", "FontSize", 15); ylabel("Loss [cm^2]", "FontWeight","bold", "FontSize", 15); legend(["Train Loss", "Valid Loss"], "FontSize",15)