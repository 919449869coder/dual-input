function lgraph = safeConnectLayers(lgraph, src, dest)
    % Check if the destination layer already has incoming connections
    destLayer = lgraph.Layers(dest);
    if ~isempty(destLayer.InputNames)
        fprintf('The destination layer %s already has incoming connections.\n', destLayer.Name);
        return;
    end

    % If the destination layer does not have incoming connections, connect the layers
    lgraph = connectLayers(lgraph, src, dest);
end
