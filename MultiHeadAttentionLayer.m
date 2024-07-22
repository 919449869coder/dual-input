classdef MultiHeadAttentionLayer < nnet.layer.Layer
    properties
        NumHeads
        EmbedDim
        QueryWeights
        KeyWeights
        ValueWeights
        OutputWeights
    end

    methods
        function layer = MultiHeadAttentionLayer(numHeads, embedDim, name)
            layer.Name = name;
            layer.NumHeads = numHeads;
            layer.EmbedDim = embedDim;
            layer.QueryWeights = randn(embedDim, embedDim);
            layer.KeyWeights = randn(embedDim, embedDim);
            layer.ValueWeights = randn(embedDim, embedDim);
            layer.OutputWeights = randn(embedDim, embedDim);
        end

        function Z = predict(layer, X)
            % Ensure input X is of the form [BatchSize, SeqLength, EmbedDim]
            [batchSize, seqLength, embedDim] = size(X);
            X = reshape(X, [], embedDim); % Reshape to [BatchSize*SeqLength, EmbedDim]

            % Compute queries, keys, and values
            Q = X * layer.QueryWeights;
            K = X * layer.KeyWeights;
            V = X * layer.ValueWeights;

            % Reshape queries, keys, and values for multi-head attention
            Q = reshape(Q, batchSize, seqLength, layer.NumHeads, embedDim / layer.NumHeads);
            K = reshape(K, batchSize, seqLength, layer.NumHeads, embedDim / layer.NumHeads);
            V = reshape(V, batchSize, seqLength, layer.NumHeads, embedDim / layer.NumHeads);

            % Compute scaled dot-product attention
            scores = sum(Q .* K, 4) / sqrt(embedDim / layer.NumHeads); % Dot product and scale
            attn = softmax(scores, 2); % Apply softmax along the sequence length dimension
            context = sum(attn .* V, 2); % Apply attention to values

            % Reshape context for output and apply output weights
            context = reshape(context, batchSize, layer.NumHeads * (embedDim / layer.NumHeads));
            Z = context * layer.OutputWeights;
        end
    end
end
