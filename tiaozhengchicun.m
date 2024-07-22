%% 加载训练集
clear; clc;

% 加载训练数据
load('traindata2345.mat');
train_data = train2345(:, 8:end);

% 创建图像数据存储器
imgsTrain = imageDatastore('train2345\kuo', 'IncludeSubfolders', true);

% 调整图像尺寸为 224x224
imgsTrainResized = transform(imgsTrain, @(x) imresize(x, [224, 224]));

% 训练集标签
train_label = categorical(train2345(:, 5));

% 创建序列数据存储器
traindata = arrayDatastore(train_data);

% 创建标签数据存储器
trainlabel = arrayDatastore(train_label);

% 合并训练数据
Train = combine(imgsTrainResized, traindata, trainlabel);

%% 加载测试集
load('testdata1.mat');
test_data = test1(:, 8:end);

% 创建图像数据存储器
imgsTest = imageDatastore('test1', 'IncludeSubfolders', true);

% 调整图像尺寸为 224x224
imgsTestResized = transform(imgsTest, @(x) imresize(x, [224, 224]));

% 测试集标签
test_label = categorical(test1(:, 5));

% 创建序列数据存储器
testdata = arrayDatastore(test_data);

% 创建标签数据存储器
testlabel = arrayDatastore(test_label);

% 合并测试数据
Test = combine(imgsTestResized, testdata, testlabel);
