#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFace检测系统使用示例
"""

import numpy as np
import cv2
import json
import time
from deepface_defense import DeepFaceDefender

def example_1_basic_detection():
    """示例1: 基础检测"""
    print("="*50)
    print("示例1: 基础检测")
    print("="*50)
    
    # 创建检测器
    defender = DeepFaceDefender()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 检测
    result = defender.detect_deepface_usage(test_image)
    
    # 显示结果
    print(f"检测结果: {result['is_deepface']}")
    print(f"置信度: {result['confidence']:.3f}")
    print(f"攻击类型: {result['attack_type']}")
    print(f"详细信息: {result['details']}")

def example_2_video_simulation():
    """示例2: 视频检测模拟"""
    print("\n" + "="*50)
    print("示例2: 视频检测模拟")
    print("="*50)
    
    defender = DeepFaceDefender()
    results = []
    
    # 模拟视频帧
    for frame_num in range(10):
        # 创建不同的测试帧
        if frame_num < 5:
            # 正常帧
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        else:
            # 可疑帧（模拟DeepFace）
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # 添加一些处理来模拟AI生成
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # 检测
        result = defender.detect_deepface_usage(frame)
        result['frame_number'] = frame_num
        results.append(result)
        
        print(f"帧 {frame_num}: {'DeepFace' if result['is_deepface'] else '正常'} "
              f"(置信度: {result['confidence']:.3f})")
    
    # 统计结果
    deepface_frames = sum(1 for r in results if r['is_deepface'])
    print(f"\n统计: 总帧数={len(results)}, DeepFace帧数={deepface_frames}")

def example_3_configuration():
    """示例3: 配置管理"""
    print("\n" + "="*50)
    print("示例3: 配置管理")
    print("="*50)
    
    from config import get_config
    
    # 获取配置
    config = get_config()
    
    # 显示配置摘要
    summary = config.get_config_summary()
    print("当前配置:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 修改配置
    config.detection.confidence_threshold = 0.8
    config.video.frame_interval = 2
    
    print(f"\n修改后配置:")
    print(f"  检测阈值: {config.detection.confidence_threshold}")
    print(f"  帧间隔: {config.video.frame_interval}")

def example_4_export_results():
    """示例4: 结果导出"""
    print("\n" + "="*50)
    print("示例4: 结果导出")
    print("="*50)
    
    defender = DeepFaceDefender()
    
    # 进行多次检测
    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        defender.detect_deepface_usage(frame)
    
    # 获取统计信息
    stats = defender.get_detection_statistics()
    print("检测统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 导出结果
    defender.export_results("example_results.json")
    print("结果已导出到 example_results.json")

def example_5_custom_threshold():
    """示例5: 自定义阈值"""
    print("\n" + "="*50)
    print("示例5: 自定义阈值")
    print("="*50)
    
    # 创建不同阈值的检测器
    thresholds = [0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        defender = DeepFaceDefender()
        defender.alert_threshold = threshold
        
        # 测试
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = defender.detect_deepface_usage(frame)
        
        print(f"阈值 {threshold}: {'报警' if result['is_deepface'] else '正常'} "
              f"(置信度: {result['confidence']:.3f})")

def main():
    """主函数"""
    print("DeepFace检测系统使用示例")
    print("本示例展示了系统的主要功能和使用方法")
    
    try:
        # 运行所有示例
        example_1_basic_detection()
        example_2_video_simulation()
        example_3_configuration()
        example_4_export_results()
        example_5_custom_threshold()
        
        print("\n" + "="*50)
        print("所有示例运行完成！")
        print("="*50)
        
    except Exception as e:
        print(f"示例运行失败: {e}")

if __name__ == "__main__":
    main()
