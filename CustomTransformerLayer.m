classdef CustomTransformerLayer < nnet.layer.Layer
    properties
        % Layer properties
        NumHeads
        EmbedDim
        FFNDim
    end
    
    properties (Learnable)
        % Layer learnable parameters
        QueryWeights
        KeyWeights
        ValueWeights
        OutputWeights
        FeedForwardWeights
        FeedForwardBias
    end
    
    methods
        function layer = CustomTransformerLayer(numHeads, embedDim, ffnDim, name)
            % Create the layer
            layer.Name = name;
            layer.NumHeads = numHeads;
            layer.EmbedDim = embedDim;
            layer.FFNDim = ffnDim;
            
            % Initialize weights
            layer.QueryWeights = randn(embedDim, embedDim);
            layer.KeyWeights = randn(embedDim, embedDim);
            layer.ValueWeights = randn(embedDim, embedDim);
            layer.OutputWeights = randn(embedDim, embedDim);
            layer.FeedForwardWeights = randn(ffnDim, embedDim);
            layer.FeedForwardBias = randn(ffnDim, 1);
        end
        
        function Z = predict(layer, X)
            % Forward pass
            % X is input data with size (batchSize, sequenceLength, embedDim)
            
            % Multi-head attention
            Q = X * layer.QueryWeights;
            K = X * layer.KeyWeights;
            V = X * layer.ValueWeights;
            AttentionScores = softmax(Q * K' / sqrt(layer.EmbedDim));
            AttentionOutput = AttentionScores * V;
            
            % Feed-forward network
            FFNOutput = relu(AttentionOutput * layer.FeedForwardWeights + layer.FeedForwardBias);
            
            % Output projection
            Z = FFNOutput * layer.OutputWeights;
        end
    end
end
