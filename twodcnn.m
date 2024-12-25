clc
clear all
%% 1读取SPA数据   1/2一个一个执行，不执行的ctrl+R注释
imgtrainspa=load('F:\耐盐\耐盐\光谱\多光谱图像\heimgalltrain.mat');
imgtrainspa=struct2cell(imgtrainspa);
imgtrain=imgtrainspa{1}';
% imgtrainspa=img1{trainspa};
label_train=xlsread('F:\耐盐\耐盐\光谱\多光谱图像\trainlabel.xlsx');
labeltrain=label_train(2:end,5);
label_test=xlsread('F:\耐盐\耐盐\光谱\多光谱图像\testlabel.xlsx');
labeltest=label_test(2:end,5);
%%  2读取全部数据  1/2一个一个执行，不执行的ctrl+R注释
imgtestspa=load('F:\耐盐\耐盐\光谱\多光谱图像\heimgalltest.mat');
imgtestspa=struct2cell(imgtestspa);
imgtest=imgtestspa{1}';
% imgtrainspa=img1{imgtrain};
% labeltrain=load('F:\df\data\训练集标签.mat');
% labeltest=load('F:\df\data\验证集标签.mat');
%%
img=imgtrain(:,1);
num = size(imgtrain,1);
for i = 1:num    

    img{i} = imresize(img{i}, [100,100],'nearest');
end
% IMGtrainlgui=img(labell(30:end),1);
% IMGvalgui=img(labell(1:29),1);

for i=1:length(img)
guizhuan90{i}=imrotate(img{i},90);
guizhuan180{i}=imrotate(img{i},180);
guizhuan270{i}=imrotate(img{i},270);
end
% 
KUOIMG=[img;guizhuan90'];
KUOIMG=[KUOIMG;guizhuan180'];
KUOIMG=[KUOIMG;guizhuan270'];
% save('C:\Users\123\Desktop\x','x');
% save('C:\Users\123\Desktop\y','y');

% yyyy=cell2mat(y);

% y = [y,y6];
% y = [y,y7];
% y = [y,y8];
% y = [y,y9];
% y = [y,y10];
% y = [y,y11];
% y = [y,y12];
% y = [y,y13];
% y = [y,y14];
% y = [y,y15];
% y = y';
% y=cell2mat(y);
% % y = categorical(y);
% save('D:\keyan\2dcnn\zhang\cho','x');
% save('D:\keyan\2dcnn\zhang\cho','y');
% 
% load('D:\keyan\2dcnn\zhang\cho');
% load('D:\keyan\2dcnn\zhang\cho');
num1= size(img,1);
% clear x
for i = 1:num1
    x(:,:,:,i) = img{i};
end
XTrain=x;
yy=labeltrain
%     ;labeltrain];
% yy=[yy;labeltrain];
% yy=[yy;labeltrain];

% idx = randperm(size(XTrain,4),100);
XVali =imgtest(:,:,:,:);

for j = 1:length(XVali )  
    XVali{j} = imresize(XVali{j}, [100,100],'nearest');
end
% clear XValidation
for k = 1:length(XVali)
    XValidation(:,:,:,k) = XVali{k};
end
% XTrain(:,:,:,idx) = [];
% YValidation = y(idx);
% y(idx) = [];

%     'RandXScale', [0.5 1], ...
%     'RandYScale', [0.5 1], ...
y=categorical(yy);
% y=y./20;
YValidation=categorical(labeltest);
% y=cell2mat(y);
imageSize = [100 100 19];

augmenter = imageDataAugmenter( ...
    'RandRotation',[0 360],...
    'RandXTranslation' ,[-3 3], ...
    'RandYTranslation' ,[-3 3])	
auimds = augmentedImageDatastore(imageSize,XTrain,y,'DataAugmentation',augmenter);


%% 训练网络

layers = [
    imageInputLayer([100 100 19])
    
    convolution2dLayer(3,512,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    % convolution2dLayer(3,512,'Padding','same')
    % batchNormalizationLayer
    % leakyReluLayer
    % maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling2dLayer(2,'Stride',2)
    % 
    % convolution2dLayer(3,256,'Padding','same')
    % batchNormalizationLayer
    % leakyReluLayer
    % maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.2)
    fullyConnectedLayer(3)          % 全连接层
    softmaxLayer
    classificationLayer];
%trainingOptions训练深度学习神经网络的选项
% options = trainingOptions('adam', ...     % SGDM 梯度下降算法
%     'MiniBatchSize', 20,...               % 批大小,每次训练样本个数30
%     'MaxEpochs',50,...                  % 最大训练次数 500
%     'InitialLearnRate', 0.01,...          % 初始学习率为0.01
%     'LearnRateSchedule', 'piecewise',...  % 学习率下降
%     'LearnRateDropFactor', 0.1,...        % 学习率下降因子 0.1
%     'LearnRateDropPeriod', 150,...        % 经过400次训练后 学习率为 0.01*0.1
%     'Shuffle', 'every-epoch',...          % 每次训练打乱数据集
%     'L2Regularization', 0.0000001, ...         % L2正则化参数
%     'Plots', 'training-progress',...      % 画出曲线
%     'Verbose', false);

options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MiniBatchSize', 20,...               % 批大小,每次训练样本个数30
    'MaxEpochs', 50, ...                  % 最大训练次数 500
    'InitialLearnRate', 1e-3, ...          % 初始学习率为 0.001
    'L2Regularization', 1e-4, ...          % L2正则化参数
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子 0.1
    'LearnRateDropPeriod', 400, ...        % 经过450次训练后 学习率为 0.001 * 0.1
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);


%%  训练模型
[net,info] = trainNetwork(auimds,layers,options)
% [net,info] = trainNetwork(p_train, t_train, layers, options);
% inputSize = net.Layers(1).InputSize;
% 
% net.Layers
figure(1)
plot(info.TrainingLoss);%%画出训练的loss
hold on
figure(2)
plot(info.TrainingAccuracy);%%画出训练的准确率

%% 模型保存
% net代表上面模型的名字
% model.mat代表保存后的模型名字和路径
save('F:\耐盐\耐盐\光谱\多光谱图像\2dcnn\model.mat', "net")
% % 请注意，代码最右侧的net一定要写，这与保存模型时，模型的名称对应
net = load("F:\耐盐\耐盐\光谱\多光谱图像\2dcnn\model.mat").net;
%% 模型评估
% 测试集
YPredtest = classify(net, p_test);
YPredtest = categorical(YPredtest);
acctest = mean(YPredtest ==t_test);

% 训练集
YPredtrain = classify(net, p_train);
YPredrain = categorical(YPredtrain);
acctrain = mean(YPredtrain ==t_train);
% 混淆矩阵
m = confusionmat(t_test,YPredtest);
% 绘制混淆矩阵
% confusionchart(m,["类别1","类别2","类别3","类别4","类别5","类别6","类别7","类别8"])
confusionchart(m,["类别1","类别2","类别3"])
A = m;
% 计算第一类的评价指标
c1_precise = m(1,1)/sum(m(:,1));
c1_recall = m(1,1)/sum(m(1,:));
c1_F1 = 2*c1_precise*c1_recall/(c1_precise+c1_recall);
% 计算第二类的评价指标
c2_precise = m(2,2)/sum(m(:,2));
c2_recall = m(2,2)/sum(m(2,:));
c2_F1 = 2*c2_precise*c2_recall/(c2_precise+c2_recall);
% 计算第三类的评价指标
c3_precise = m(3,3)/sum(m(:,3));
c3_recall = m(3,3)/sum(m(3,:));
c3_F1 = 2*c3_precise*c3_recall/(c3_precise+c3_recall);
% % 计算第四类的评价指标
% c4_precise = m(4,4)/sum(m(:,4));
% c4_recall = m(4,4)/sum(m(4,:));
% c4_F1 = 2*c4_precise*c4_recall/(c4_precise+c4_recall);
% % 计算第五类的评价指标
% c5_precise = m(5,5)/sum(m(:,5));
% c5_recall = m(5,5)/sum(m(5,:));
% c5_F1 = 2*c5_precise*c5_recall/(c5_precise+c5_recall);
% % 计算第六类的评价指标
% c6_precise = m(6,6)/sum(m(:,6));
% c6_recall = m(6,6)/sum(m(6,:));
% c6_F1 = 2*c6_precise*c6_recall/(c6_precise+c6_recall);
% % 计算第七类的评价指标
% c7_precise = m(7,7)/sum(m(:,7));
% c7_recall = m(7,7)/sum(m(7,:));
% c7_F1 = 2*c7_precise*c7_recall/(c7_precise+c7_recall);

macroPrecise = (c1_precise+c2_precise+c3_precise)/3;
macroRecall = (c1_recall+c2_recall+c3_recall)/3;
macroF1 = (c1_F1+c2_F1+c3_F1)/3;
