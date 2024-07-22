clc
clear all
% 假设 hs_image_cell 是包含761个多光谱图像的cell数组
load('I:\耐盐\耐盐\光谱\SPECTRALSHUJU\图像序列\wuzhe\train1345.mat')
img=train1345(:,1);
img=test5tezheng';
num = size(img,1);
% for i = 1:num    
% 
%     img{i} = imresize(img{i}, [100,100],'nearest');
% end
% % IMGtrainlgui=img(labell(30:end),1);
% % IMGvalgui=img(labell(1:29),1);
% 
% for i=1:length(img)
% guizhuan90{i}=imrotate(img{i},90);
% guizhuan180{i}=imrotate(img{i},180);
% guizhuan270{i}=imrotate(img{i},270);
% end
% 
% KUOIMG=[img;guizhuan90'];
% KUOIMG=[KUOIMG;guizhuan180'];
% KUOIMG=[KUOIMG;guizhuan270'];
% 构造保存文件名的基本部分
base_filename = 'multi_channel_image';
% numel(test)
% 循环处理每个图像
% KUOIMG=img(1:10);
for i = 1:numel(img)

    % 获取当前图像数据
    hs_image = img{i};
    
    % 将数据范围从 [0, 1] 映射到 [0, 255]，并转换为 uint8 类型
    min_value = min(hs_image(:));
    max_value = max(hs_image(:));
    normalized_hs_image = (hs_image - min_value) / (max_value - min_value);
    hs_image_uint8 = uint8(normalized_hs_image * 255);
    
    % 构造保存文件名
    filename = sprintf('%s_00%03d.tif', base_filename, i);

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
end
