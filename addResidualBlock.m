function lgraph = addResidualBlock(lgraph, numFilters, layerName)
    % 添加残差块到图层中
    layers = [
        convolution2dLayer([3 3], numFilters, 'Name', [layerName, '_branch2a'], 'Padding', [1 1 1 1])
        batchNormalizationLayer('Name', ['bn', layerName, '_branch2a'])
        reluLayer('Name', [layerName, '_branch2a_relu'])
        convolution2dLayer([3 3], numFilters, 'Name', [layerName, '_branch2b'], 'Padding', [1 1 1 1])
        batchNormalizationLayer('Name', ['bn', layerName, '_branch2b'])
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, [layerName, '_branch2b'], [layerName, '_branch2a_relu']);
    lgraph = connectLayers(lgraph, [layerName, '_branch2a_relu'], [layerName, '_branch2b']);
end
