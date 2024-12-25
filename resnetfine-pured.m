%% 混淆矩阵
% 从已训练的网络中提取层
layers = trainedNet.Layers;
connections = trainedNet.Connections;

% 初始化剪枝后的层数组
prunedLayers = layers;

% 设置动态剪枝参数
pruneCoeff = 0.5; % 剪枝系数
pruningInfo={};
% 剪枝
for i = 1:numel(layers)
    if isa(layers(i), 'nnet.cnn.layer.Convolution2DLayer') && layers(i).NumFilters > 1
        weights = layers(i).Weights;
        bias = layers(i).Bias;
        
        % 计算权重标准差
        weightStd = std(weights(:));
        
        % 根据权重标准差设置动态剪枝阈值
        pruneThreshold = pruneCoeff * weightStd;
        
        % 剪枝：将小于阈值的权重置为零
        prunedIndices = abs(weights) < pruneThreshold;
        weights(prunedIndices) = 0;
        
        % 移除零权重的通道
        nonZeroFilters = any(weights ~= 0, [1, 2, 3]);
        numPrunedFilters = layers(i).NumFilters - sum(nonZeroFilters);
        
        % 记录剪枝信息
        pruningInfo = [pruningInfo; {layers(i).Name, layers(i).NumFilters, sum(nonZeroFilters), numPrunedFilters}];
        
        % 重新创建剪枝后的卷积层
        newConvLayer = convolution2dLayer(layers(i).FilterSize, sum(nonZeroFilters), ...
                                          'Stride', layers(i).Stride, ...
                                          'Padding', layers(i).Padding, ...
                                          'Name', layers(i).Name);
        newConvLayer.Weights = weights(:, :, :, nonZeroFilters);
        newConvLayer.Bias = bias(nonZeroFilters);
        
        % 替换原始层
        prunedLayers(i) = newConvLayer;
    end
end

% 创建一个新的 layerGraph
lgraph_pruned = layerGraph();

% 添加剪枝后的层到新的 layerGraph
for i = 1:numel(prunedLayers)
    lgraph_pruned = addLayers(lgraph_pruned, prunedLayers(i));
end

% 重新连接层
for i = 1:size(connections, 1)
    lgraph_pruned = connectLayers(lgraph_pruned, connections.Source{i}, connections.Destination{i});
end

% 显示剪枝信息
disp('Pruning Information:');
disp('Layer Name | Original Filters | Remaining Filters | Pruned Filters');

% 将表格写入 Excel 文件
% 创建剪枝信息的表格
pruningTable = cell2table(pruningInfo, 'VariableNames', {'LayerName', 'OriginalFilters', 'RemainingFilters', 'PrunedFilters'});

% 将表格写入 Excel 文件
pruningTable = cell2table(pruningInfo, 'VariableNames', {'LayerName', 'OriginalFilters', 'RemainingFilters', 'PrunedFilters'});
excelFileName = 'pruning_info.xlsx'; % Excel 文件名
writetable(pruningTable, excelFileName);

% 新的输出通道数
newNumFilters = 128;

% 替换原始卷积层，同时将输出通道数更改为128
lgraph_pruned = replaceLayer(lgraph_pruned, 'res3a_branch2a', convolution2dLayer([3,3], newNumFilters, 'Name', 'res3a_branch2a'));

% 定义池化层的尺寸和步幅
poolSize = [2, 2]; % 设置池化层的大小为 2x2
stride = 2; % 设置步幅为 2

% 添加平均池化层，并设置池化层的参数
poolingLayer = averagePooling2dLayer(poolSize, 'Name', 'poolingLayer', 'Stride', stride);


% 添加池化层到图中，并连接到 bn3a_branch2b 层之后
lgraph_pruned = addLayers(lgraph_pruned, poolingLayer);
lgraph_pruned = connectLayers(lgraph_pruned, 'bn3a_branch2b', 'poolingLayer');

% 添加1x1卷积层，调整通道数和尺寸
numChannels = 128; % 设置输出通道数为 1
convLayer = convolution2dLayer(1, numChannels, 'Name', 'conv_1x1');

% 添加卷积层到图中，并连接到池化层之后
lgraph_pruned = addLayers(lgraph_pruned, convLayer);
lgraph_pruned = connectLayers(lgraph_pruned, 'poolingLayer', 'conv_1x1');


% 将 conv_1x1 的输出连接到 res_3a 的输入
% 断开 res3a 的第一个输入连接

lgraph_pruned = disconnectLayers(lgraph_pruned, "bn3a_branch2b","res3a/in1");

% 将 conv_1x1 的输出连接到 res3a 的第一个输入
lgraph_pruned = connectLayers(lgraph_pruned, 'conv_1x1', 'res3a/in1');


% 定义池化层的尺寸和步幅
poolSize1 = [2, 2]; % 设置池化层的大小为 2x2
stride1 = 1; % 设置步幅为 2

% 添加平均池化层，并设置池化层的参数
poolingLayer1 = averagePooling2dLayer(poolSize1, 'Name', 'poolingLayer1', 'Stride', stride1);


% 添加池化层到图中，并连接到 bn3a_branch2b 层之后
lgraph_pruned = addLayers(lgraph_pruned, poolingLayer1);
lgraph_pruned = connectLayers(lgraph_pruned, 'bn3a_branch1', 'poolingLayer1');

% 断开 res3a 的第一个输入连接

lgraph_pruned = disconnectLayers(lgraph_pruned, "bn3a_branch1","res3a/in2");

% 将 conv_1x1 的输出连接到 res3a 的第一个输入
lgraph_pruned = connectLayers(lgraph_pruned, 'poolingLayer1', 'res3a/in2');
% 新的输出通道数
newNumFilters = 256;

% 替换原始卷积层，同时将输出通道数更改为128
lgraph_pruned = replaceLayer(lgraph_pruned, 'res4a_branch2b', convolution2dLayer([3,3], newNumFilters, 'Name', 'res4a_branch2b'));
poolSize2 = [3, 3]; % 设置池化层的大小为 2x2
stride2 = 1; % 设置步幅为 2
% 添加平均池化层，并设置池化层的参数
poolingLayer2 = averagePooling2dLayer(poolSize2, 'Name', 'poolingLayer2', 'Stride', stride2);
% 添加池化层到图中，并连接到 bn4a_branch1层之后
lgraph_pruned = addLayers(lgraph_pruned, poolingLayer2);
lgraph_pruned = connectLayers(lgraph_pruned, 'bn4a_branch1', 'poolingLayer2');


% 断开 res3a 的第一个输入连接

lgraph_pruned = disconnectLayers(lgraph_pruned, "bn4a_branch1","res4a/in2");

% 将 conv_1x1 的输出连接到 res3a 的第一个输入
lgraph_pruned = connectLayers(lgraph_pruned, 'poolingLayer2', 'res4a/in2');
newNumFilters = 256;

% 替换原始卷积层，同时将输出通道数更改为128
lgraph_pruned = replaceLayer(lgraph_pruned, 'res4b_branch2a', convolution2dLayer([3,3], newNumFilters, 'Name', 'res4b_branch2a'));

poolSize3 = [3, 3]; % 设置池化层的大小为 2x2
stride3 = 1; % 设置步幅为 2
% 添加平均池化层，并设置池化层的参数
poolingLayer3 = averagePooling2dLayer(poolSize3, 'Name', 'poolingLayer3', 'Stride', stride3);
% 添加池化层到图中，并连接到 bn4a_branch1层之后
lgraph_pruned = addLayers(lgraph_pruned, poolingLayer3);
lgraph_pruned = connectLayers(lgraph_pruned, 'res4a_relu', 'poolingLayer3');


% 断开 res3a 的第一个输入连接

lgraph_pruned = disconnectLayers(lgraph_pruned, "res4a_relu","res4b/in2");

% 将 conv_1x1 的输出连接到 res3a 的第一个输入
lgraph_pruned = connectLayers(lgraph_pruned, 'poolingLayer3', 'res4b/in2');
% newNumFilters = 256;
%% 找到要删除的层的名称
layersToRemove = {'bn5a_branch2a', 'res5a_branch2a_relu','bn5a_branch1','bn5a_branch2b'};

% 从图中删除指定的层
lgraph_pruned = removeLayers(lgraph_pruned, layersToRemove);
lgraph_pruned  = connectLayers(lgraph_pruned ,"res5a_branch2a","res5a_branch2b");
lgraph_pruned  = connectLayers(lgraph_pruned ,"res5a_branch1","res5a/in2");


newNumFilters = 512;

% 替换原始卷积层，同时将输出通道数更改为128
lgraph_pruned = replaceLayer(lgraph_pruned, 'res5a_branch2a', convolution2dLayer([3,3], newNumFilters, 'Name', 'res5a_branch2a'));
% poolSize4 = [4, 4]; % 设置池化层的大小为 2x2
% stride4 = 1; % 设置步幅为 2
% % 添加平均池化层，并设置池化层的参数
% poolingLayer4 = averagePooling2dLayer(poolSize4, 'Name', 'poolingLayer4', 'Stride', stride4);
% % 添加池化层到图中，并连接到 bn4a_branch1层之后
% lgraph_pruned = addLayers(lgraph_pruned, poolingLayer4);
% lgraph_pruned = connectLayers(lgraph_pruned, 'res5a_branch2b', 'poolingLayer4');
% 
% % 将 conv_1x1 的输出连接到 res3a 的第一个输入
lgraph_pruned = connectLayers(lgraph_pruned, 'res5a_branch2b', 'res5a/in1');
newNumFilters = 345;

% 替换原始卷积层，同时将输出通道数更改为128
lgraph_pruned = replaceLayer(lgraph_pruned, 'res5a_branch1', convolution2dLayer([3,3], newNumFilters, 'Name', 'res5a_branch1'));
newNumFilters = 345;

% 替换原始卷积层，同时将输出通道数更改为128
lgraph_pruned = replaceLayer(lgraph_pruned, 'res5b_branch2a', convolution2dLayer([3,3], newNumFilters, 'Name', 'res5b_branch2a'));
% % 替换原始卷积层，同时将输出通道数更改为128
% lgraph_pruned = replaceLayer(lgraph_pruned, 'res4b_branch2a', convolution2dLayer([3,3], newNumFilters, 'Name', 'res4b_branch2a'));
layersToRemove = {'bn5b_branch2a', 'res5b_branch2a_relu','bn5b_branch2b'};

% 从图中删除指定的层
lgraph_pruned = removeLayers(lgraph_pruned, layersToRemove);
lgraph_pruned  = connectLayers(lgraph_pruned ,"res5b_branch2a","res5b_branch2b");
lgraph_pruned  = connectLayers(lgraph_pruned ,"res5b_branch2b","res5b/in1");
newNumFilters = 345;

% 替换原始卷积层，同时将输出通道数更改为128
lgraph_pruned = replaceLayer(lgraph_pruned, 'res5b_branch2b', convolution2dLayer([3,3], newNumFilters, 'Name', 'res5b_branch2b'));
% 显示更新后的网络结构

poolSize5 = [1, 1]; % 设置池化层的大小为 2x2
stride5 = 2; % 设置步幅为 2
% 添加平均池化层，并设置池化层的参数
poolingLayer5 = averagePooling2dLayer(poolSize5, 'Name', 'poolingLayer5', 'Stride', stride5);
% 添加池化层到图中，并连接到 bn4a_branch1层之后
lgraph_pruned = addLayers(lgraph_pruned, poolingLayer5);
lgraph_pruned = connectLayers(lgraph_pruned, 'res5a_relu', 'poolingLayer5');
lgraph_pruned = disconnectLayers(lgraph_pruned, "res5a_relu","res5b/in2");

% 将 conv_1x1 的输出连接到 res3a 的第一个输入
lgraph_pruned = connectLayers(lgraph_pruned, 'poolingLayer5', 'res5b/in2');
% 移除后四层
% 移除指定层
layers_to_remove = {'concat', 'fc', 'softmax', 'classoutput'};
lgraph_pruned = removeLayers(lgraph_pruned, layers_to_remove);


% 添加第三个网络分支的层
tempLayers3 = [
    concatenationLayer(1, 2, 'Name', 'concat')
    fullyConnectedLayer(3, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];
lgraph_pruned = addLayers(lgraph_pruned, tempLayers3);

% 连接第一个分支
lgraph_pruned = connectLayers(lgraph_pruned, 'flatten', 'concat/in1');

% 连接第二个分支的LSTM输出
lgraph_pruned = connectLayers(lgraph_pruned, 'lstm', 'concat/in2');

% 显示新网络结构
analyzeNetwork(lgraph_pruned);



% 构建微调模型
lgraph_finetune = lgraph_pruned;
disp(pruningInfo);
% 微调参数
miniBatchSize = 16;
validationFrequency = 50;
options = trainingOptions('sgdm', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 50, ...
    'InitialLearnRate', 0.01, ...
    'L2Regularization', 1e-4, ...          
    'ValidationData', Test, ...
    'ValidationFrequency', validationFrequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');%adam



% 微调模型
prunedNet = trainNetwork(Train, lgraph_finetune, options);
y_pred_train = classify(prunedNet,Train);
accuracy_train = mean(y_pred_train == train_label);
disp(['微调后训练集准确率：',num2str(100*accuracy_train),'%']);
%% 混淆矩阵
figure(1)
plotconfusion(train_label, y_pred_train);
%%x测试
y_pred_test = classify(prunedNet,Test);
accuracy_test = mean(y_pred_test == test_label);
disp(['微调后测试集准确率：',num2str(100*accuracy_test),'%']);
