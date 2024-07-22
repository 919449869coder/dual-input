%% 加载数据
% 训练集图像是160异常+160正常, 测试集图像是40异常+40正常
% train_data里, 是160列异常+160列正常, test_data里, 是40异常+40正常 图像和序列一一对应
%% 加载训练集
clear; clc;
% load('train_data.mat'); % 一列是一个样本，每个样本500个数据点，320本个样
%      
load('traindata2345.mat');
% train_datadata=[train1345;train1345];
% train_datadata=[train_datadata;train1345];
% train_datadata=[train_datadata;train1345];
 train_datadata=train2345;
train_data=train_datadata(1:end,8:end);

traindata = arrayDatastore(train_data); % 训练集序列：得先弄成一行是一个样本，再转化成arrayDatastore

imgsTrain = imageDatastore('train2345\kuo','IncludeSubfolders' ,true);% 训练集图像

%% 训练集标签
train_label = categorical(train_datadata(1:end,5)); % 训练集标签：0对应异常，1对应正常
trainlabel = arrayDatastore(train_label); % 得先是一列categorical，再转化

% 合并训练集
Train = combine(imgsTrain,traindata,trainlabel);%% 图像序列合并

%% 加载测试集
load('testdata1.mat');
test_data=test1(1:end,8:end);
testdata = arrayDatastore(test_data); % 测试集序列：得先弄成一行是一个样本，再转化 正常是序列是arrayDatastore，图像是imgsDatastore
imgsTest = imageDatastore('test1','IncludeSubfolders' ,true);% 训练集图像
test_label = categorical(test1(1:end,5)); % 训练集标签：0对应异常，1对应正常
testlabel = arrayDatastore(test_label); % 得先是一列categorical，再转化
Test = combine(imgsTest,testdata,testlabel);

%% 构建双输入网络，一边是序列输入，一边是图像输入
lgraph = layerGraph();

% 添加层分支 将网络分支添加到层次图中
tempLayers = [
    imageInputLayer([100 100 19],"Name","图像输入") % 图像输入尺寸224*224*3
    convolution2dLayer([3 3],8,"Name","conv") % 3*3卷积，8个
    reluLayer("Name","relu")
    maxPooling2dLayer([11 11],"Name","maxpool","Stride",[11 11]) % 11*11池化，步长11
    convolution2dLayer([3 3],4,"Name","conv_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([3 3],"Name","maxpool_2","Stride",[2 2])
    flattenLayer("Name","flatten")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers1 = [
    imageInputLayer([1 39 1],"Name","序列输入") % 序列长度1*500*1，这里是把序列当成1*500*1的图像输入，这样跟一维卷积没有任何区别
    convolution2dLayer([1 7],8,"Name","conv_1") % 1*7卷积，8个
    reluLayer("Name","relu_1")
    % maxPooling2dLayer([1 64],"Name","maxpool_1","Stride",[64 64]) % 1*64池化，步长64
    flattenLayer("Name","flatten_1")];
lgraph = addLayers(lgraph,tempLayers1);


tempLayers2 = [
    concatenationLayer(1,2,"Name","concat") % 拼接层
    fullyConnectedLayer(3,"Name","fc") % 全连接，2代表2分类
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers2);

% 清理辅助变量
clear tempLayers;

% 连接层分支, 连接网络的所有分支以创建网络图。
lgraph = connectLayers(lgraph,"flatten_1","concat/in2");
lgraph = connectLayers(lgraph,"flatten","concat/in1");

% 显示网络结构
analyzeNetwork(lgraph);

% 构建优化器
options = trainingOptions('sgdm',... % 随机梯度下降法
    'MiniBatchSize',5,... % 批大小
    'MaxEpochs',50,... % 轮数
    'InitialLearnRate',0.01,... % 学习率
    'Verbose',1,...
    'ExecutionEnvironment','auto',... % 自动选CPU/GPU运行
    'Plots','training-progress');

%% 训练

trainedShuangNet = trainNetwork(Train,lgraph,options);%p_train, t_train 合并好的训练
% save('2dcnnshuangmodel13542-bukuo.mat', "trainedShuangNet")
%trainedShuangNet = trainNetwork([XTrain;p_train],train_label,lgraph,options);%p_train, t_train
%%x训练
y_pred_train = classify(trainedShuangNet,Train);
accuracy_train = mean(y_pred_train == train_label);
disp(['训练集准确率：',num2str(100*accuracy_train),'%']);
%% 混淆矩阵
figure(1)
plotconfusion(train_label, y_pred_train);
%%x测试
y_pred_test = classify(trainedShuangNet,Test);
accuracy_test = mean(y_pred_test == test_label);
disp(['测试集准确率：',num2str(100*accuracy_test),'%']);
%% 混淆矩阵
figure(2)
plotconfusion(test_label, y_pred_test);


%%  混淆矩阵
figure
cm = confusionchart(train_label, y_pred_train);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(test_label, y_pred_test);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

m = confusionmat(test_label, y_pred_test);
% 绘制混淆矩阵
% confusionchart(m,["类别1","类别2","类别3"])
% confusionchart(m,["Ⅰ","Ⅱ","Ⅲ"])
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

% T(:,1)=T_train';
% T(:,2)=T_sim1';
% T(1:length(T_test),3)=T_test';
% T(1:length(T_test),4)=T_sim2';

% % 找到全连接层的索引
% fc_layer_index = find(arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.FullyConnectedLayer'), trainedShuangNet.Layers));
% 
% % 获取全连接层的权重
% weights = trainedShuangNet.Layers(fc_layer_index).Weights;
% 
% % 可视化权重矩阵
% figure;
% imagesc(weights);
% colorbar;
% title('Fully Connected Layer Weights');
% xlabel('输入特征');
% ylabel('输出特征');
% 
% 
% % 找到全连接层的索引
% fc_layer_index = find(arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.FullyConnectedLayer'), trainedShuangNet.Layers));
% 
% % 获取全连接层的权重
% weights = trainedShuangNet.Layers(fc_layer_index).Weights;
% 
% % 绘制全连接层网络
% figure;
% colormap jet; % 设置颜色映射
% plot(weights, 'LineWidth', 2); % 绘制连接权重
% title('Fully Connected Layer Network'); % 设置图标题
% xlabel('Input Neuron'); % 设置X轴标签
% ylabel('Output Neuron'); % 设置Y轴标签
% colorbar; % 显示颜色条
% fc_layer_index = find(arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.FullyConnectedLayer'), trainedShuangNet.Layers));
% 
% % 获取全连接层的输出大小
% output_size = trainedShuangNet.Layers(fc_layer_index).OutputSize;
% 
% % 显示全连接层的输出大小
% disp('全连接层的输出大小：');
% disp(output_size);
% 
% fc_layer_index = find(arrayfun(@(layer) isa(layer, 'nnet.cnn.layer.FullyConnectedLayer'), trainedShuangNet.Layers));
% 
% % 获取全连接层的偏置
% biases = trainedShuangNet.Layers(fc_layer_index).Bias;
% 
% % 显示偏置值
% disp('全连接层的偏置值：');
