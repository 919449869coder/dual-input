function transformerLayer = createTransformerLayer(num_heads, d_model, dff, dropout)
    % Multi-head self-attention layer
    self_attention = multiHeadSelfAttentionLayer(num_heads, 'NumHeads', num_heads, 'QueryDepth', d_model, 'ValueDepth', d_model, 'OutputDepth', d_model, 'AttentionScale', sqrt(d_model), 'Name', 'multi_head_self_attention');
    norm1 = layerNormalizationLayer('Epsilon', 1e-6, 'Name', 'self_attention_norm');
    dropout1 = dropoutLayer(dropout, 'Name', 'self_attention_dropout');
    skip1 = additionLayer(2, 'Name', 'self_attention_residual');

    % Feedforward neural network layer
    feedforward_nn = [
        fullyConnectedLayer(dff, 'Name', 'ffn_fc_1')
        reluLayer('Name', 'ffn_relu')
        fullyConnectedLayer(d_model, 'Name', 'ffn_fc_2')
      ];
    feedforward = [
        dnnLayers % Add deep learning layers
        additionLayer(2, 'Name', 'ffn_residual')
      ];
    norm2 = layerNormalizationLayer('Epsilon', 1e-6, 'Name', 'ffn_norm');
    dropout2 = dropoutLayer(dropout, 'Name', 'ffn_dropout');
    skip2 = additionLayer(2, 'Name', 'ffn_residual_connection');
    
    % Assemble the layers into a layerGraph
    transformerLayer = layerGraph([
        imageInputLayer([1 1 d_model], 'Name', 'Input') % Input layer
        self_attention
        norm1
        dropout1
        skip1
        feedforward
        norm2
        dropout2
        skip2
    ]);
end