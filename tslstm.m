%% 加载数据
clear; clc;

% 加载训练集
load('traindata2345.mat');
train_datadata = train2345;
train_data = train_datadata(:, 8:end);

traindata = arrayDatastore(train_data);
imgsTrain = imageDatastore('train2345\kuo', 'IncludeSubfolders', true);

% 加载训练集标签
train_label = categorical(train_datadata(:, 5));
trainlabel = arrayDatastore(train_label);

% 合并训练集
Train = combine(imgsTrain, traindata, trainlabel);

% 加载测试集
load('testdata1.mat');
test_data = test1(:, 8:end);
testdata = arrayDatastore(test_data);
imgsTest = imageDatastore('test1', 'IncludeSubfolders', true);

% 加载测试集标签
test_label = categorical(test1(:, 5));
testlabel = arrayDatastore(test_label);

% 合并测试集
Test = combine(imgsTest, testdata, testlabel);


%% 构建双输入网络
lgraph = layerGraph();

% 图像输入部分（使用预训练的ResNet50）
net = resnet50();
imgInputLayer = imageInputLayer([100 100 19], "Name", "图像输入"); % 根据您的图像大小调整输入层尺寸
lgraph = addLayers(lgraph, imgInputLayer);

% 移除ResNet50的最后三层（全连接层、Softmax层、分类层）
layers = layerGraph(net);
layers = removeLayers(layers, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

% 连接ResNet50
lgraph = addLayers(lgraph, layers);
lgraph = connectLayers(lgraph, '图像输入', 'conv1');

% 添加新的全连接层、softmax层和分类层
imgFeatureLayers = [
    fullyConnectedLayer(128, "Name", "fc_img")
    reluLayer("Name", "relu_img")
    dropoutLayer(0.5, "Name", "dropout_img")
    flattenLayer("Name", "flatten_img")];
lgraph = addLayers(lgraph, imgFeatureLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'fc_img');

% 序列输入部分（加LSTM和Transformer）
seqLayers = [
    imageInputLayer([1 39 1], "Name", "序列输入")
    convolution2dLayer([1 7], 8, "Name", "conv_seq")
    reluLayer("Name", "relu_seq")
    flattenLayer("Name", "flatten_seq")
    lstmLayer(50, "OutputMode", "last", "Name", "lstm")
    transformerEncoderLayer(64, 4, 128, "Name", "transformer")
    flattenLayer("Name", "flatten_seq2")];
lgraph = addLayers(lgraph, seqLayers);

% 合并层
mergeLayers = [
    concatenationLayer(1, 2, "Name", "concat")
    fullyConnectedLayer(3, "Name", "fc")
    softmaxLayer("Name", "softmax")
    classificationLayer("Name", "classoutput")];
lgraph = addLayers(lgraph, mergeLayers);

% 连接层
lgraph = connectLayers(lgraph, "flatten_img", "concat/in1");
lgraph = connectLayers(lgraph, "flatten_seq2", "concat/in2");

% 显示网络结构
analyzeNetwork(lgraph);

%% 构建优化器
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 5, ...
    'MaxEpochs', 50, ...
    'InitialLearnRate', 0.01, ...
    'Verbose', 1, ...
    'ExecutionEnvironment', 'auto', ...
    'Plots', 'training-progress');

%% 训练网络
trainedShuangNet = trainNetwork(Train, lgraph, options);

%% 评估模型
% 训练集评估
y_pred_train = classify(trainedShuangNet, Train);
accuracy_train = mean(y_pred_train == train_label);
disp(['训练集准确率：', num2str(100 * accuracy_train), '%']);
figure(1);
plotconfusion(train_label, y_pred_train);

% 测试集评估
y_pred_test = classify(trainedShuangNet, Test);
accuracy_test = mean(y_pred_test == test_label);
disp(['测试集准确率：', num2str(100 * accuracy_test), '%']);
figure(2);
plotconfusion(test_label, y_pred_test);

%% 绘制混淆矩阵
figure;
cm_train = confusionchart(train_label, y_pred_train);
cm_train.Title = 'Confusion Matrix for Train Data';
cm_train.ColumnSummary = 'column-normalized';
cm_train.RowSummary = 'row-normalized';

figure;
cm_test = confusionchart(test_label, y_pred_test);
cm_test.Title = 'Confusion Matrix for Test Data';
cm_test.ColumnSummary = 'column-normalized';
cm_test.RowSummary = 'row-normalized';

%% 计算评价指标
m = confusionmat(test_label, y_pred_test);

c1_precise = m(1, 1) / sum(m(:, 1));
c1_recall = m(1, 1) / sum(m(1, :));
c1_F1 = 2 * c1_precise * c1_recall / (c1_precise + c1_recall);

c2_precise = m(2, 2) / sum(m(:, 2));
c2_recall = m(2, 2) / sum(m(2, :));
c2_F1 = 2 * c2_precise * c2_recall / (c2_precise + c2_recall);

c3_precise = m(3, 3) / sum(m(:, 3));
c3_recall = m(3, 3) / sum(m(3, :));
c3_F1 = 2 * c3_precise * c3_recall / (c3_precise + c3_recall);

macroPrecise = (c1_precise + c2_precise + c3_precise) / 3;
macroRecall = (c1_recall + c2_recall + c3_recall) / 3;
macroF1 = (c1_F1 + c2_F1 + c3_F1) / 3;

disp(['Macro Precision: ', num2str(macroPrecise)]);
disp(['Macro Recall: ', num2str(macroRecall)]);
disp(['Macro F1 Score: ', num2str(macroF1)]);
