function lgraph = addLSTMLayer(lgraph, numHiddenUnits, layerName)
    % 添加 LSTM 层到图层中
    layers = [
        sequenceInputLayer([1 39], 'Name', [layerName, '_input']) % 序列输入
        lstmLayer(numHiddenUnits, 'Name', [layerName, '_lstm']) % LSTM 层
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, [layerName, '_input'], [layerName, '_lstm']);
end
