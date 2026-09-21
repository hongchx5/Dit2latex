#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片黑白反转工具
读取inputdir里的所有图片，然后将其黑白反转，保存到output_dir中
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np


def invert_image_colors(image_path, output_path):
    """
    对单张图片进行黑白反转
    """
    # 打开图片
    img = Image.open(image_path)
    
    # 转换为numpy数组
    img_array = np.array(img)
    
    # 如果是灰度图 (2D array) 或 RGB 图 (3D array with 3 channels)
    if len(img_array.shape) == 2:  # 灰度图
        inverted_array = 255 - img_array
    elif len(img_array.shape) == 3:  # 彩色图
        if img_array.shape[2] == 3:  # RGB
            inverted_array = 255 - img_array
        elif img_array.shape[2] == 4:  # RGBA
            # 分离RGB和Alpha通道
            rgb_channels = 255 - img_array[:, :, :3]  # 反转RGB
            alpha_channel = img_array[:, :, 3]  # 保留Alpha通道
            inverted_array = np.concatenate([rgb_channels, alpha_channel[:, :, np.newaxis]], axis=2)
        else:
            raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")
    
    # 转换回PIL Image对象
    inverted_img = Image.fromarray(inverted_array.astype(np.uint8))
    
    # 保存到输出路径
    inverted_img.save(output_path)


def process_directory(input_dir, output_dir, supported_formats=None):
    """
    处理整个目录中的图片
    """
    if supported_formats is None:
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif')
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计处理的文件数
    processed_count = 0
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # print(f"正在处理: {filename}")
            try:
                invert_image_colors(input_path, output_path)
                # print(f"已保存: {output_path}")
                processed_count += 1
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
    
    print(f"\n处理完成！共处理了 {processed_count} 张图片。")


def main():    
    # 处理图片
    input_dir = './test'
    output_dir = './test1'
    os.makedirs(output_dir, exist_ok=True)
    process_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()