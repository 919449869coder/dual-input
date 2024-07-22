% % 假设hs_image是一个19通道的多光谱图像
% % 假设通道数据已经归一化到0-1范围内
% 
% % 构造保存文件名
% [num_rows, num_cols, num_channels] = size(hs_image_cell);
% data = reshape(hs_image_cell, [num_rows * num_cols, num_channels]);
% % 进行主成分分析
% coeff = pca(data);
% 
% % 选择前三个主成分
% coeff_3d = coeff(:, 1:3);
% % 投影数据到三维空间
% rgb_data = data * coeff_3d;
% % 将数据归一化到0-1范围内
% rgb_data = mat2gray(rgb_data);
% % 将数据重新排列为图像尺寸
% rgb_image = reshape(rgb_data, [num_rows, num_cols, 3]);
% 
% % 保存为RGB图像
% imwrite(rgb_image, 'pca_rgb_image.png');


% 假设 hs_image 是一个多光谱图像，数据范围在 [0, 1] 之间
% 将数据范围从 [0, 1] 映射到 [0, 255]，并转换为 uint8 类型


for i=1:size(hs_image_cell,3)
hs_image=hs_image_cell(:,:,i);
min_value = min(hs_image(:));
max_value = max(hs_image(:));
normalized_hs_image(:,:,i) = (hs_image - min_value) / (max_value - min_value);
hs_image_uint8 = uint8(normalized_hs_image * 255);
end
imshow(hs_image_uint8(:,:,19))
% 构造保存文件名
% 假设 hs_image_uint8 是你的多光谱图像数据，已经转换为 uint8 类型

% 构造保存文件名
filename = 'multi_channel_image.tif';

% 打开一个 Tiff 文件以写入
t = Tiff(filename, 'w');

% 设置 TIFF 文件的一些属性
tagstruct.ImageLength = size(hs_image_uint8, 1);
tagstruct.ImageWidth = size(hs_image_uint8, 2);
tagstruct.Compression = Tiff.Compression.None; % 可以选择压缩方式
tagstruct.Photometric = Tiff.Photometric.MinIsBlack; % 图像类型
tagstruct.BitsPerSample = 8; % 每个通道的位深度
tagstruct.SamplesPerPixel = size(hs_image_uint8, 3); % 通道数
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; % 通道的排列方式

% 写入图像数据
t.setTag(tagstruct);
t.write(hs_image_uint8);

% 关闭 Tiff 文件
t.close();
