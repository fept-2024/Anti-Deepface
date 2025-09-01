#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFace检测和防御系统主程序
"""

import sys
import os
import argparse
from typing import Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepface_defense import DeepFaceDefender
from video_detector import VideoDeepFaceDetector
from realtime_detector import RealtimeDetector
from config import get_config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DeepFace检测和防御系统')
    parser.add_argument('--mode', choices=['video', 'realtime', 'test'], 
                       default='test', help='运行模式')
    parser.add_argument('--input', '-i', type=str, help='输入视频文件路径')
    parser.add_argument('--output', '-o', type=str, default='results.json', 
                       help='输出结果文件路径')
    parser.add_argument('--threshold', '-t', type=float, default=0.7, 
                       help='检测置信度阈值')
    parser.add_argument('--camera', '-c', type=int, default=0, 
                       help='摄像头ID（实时模式）')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config()
    
    print("="*60)
    print("DeepFace检测和防御系统")
    print("="*60)
    print(f"运行模式: {args.mode}")
    print(f"检测阈值: {args.threshold}")
    
    if args.mode == 'video':
        run_video_detection(args, config)
    elif args.mode == 'realtime':
        run_realtime_detection(args, config)
    elif args.mode == 'test':
        run_test_detection(config)
    else:
        print("未知的运行模式")
        sys.exit(1)

def run_video_detection(args, config):
    """运行视频检测"""
    if not args.input:
        print("错误: 视频模式需要指定输入文件路径 (--input)")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    print(f"开始检测视频: {args.input}")
    
    try:
        detector = VideoDeepFaceDetector()
        report = detector.detect_video(args.input, args.output)
        detector.print_summary(report)
        
        print(f"\n检测完成！结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"视频检测失败: {e}")
        sys.exit(1)

def run_realtime_detection(args, config):
    """运行实时检测"""
    print(f"启动实时检测，摄像头ID: {args.camera}")
    print("按 'q' 键退出检测")
    
    try:
        detector = RealtimeDetector(args.camera)
        detector.start_detection()
        
    except Exception as e:
        print(f"实时检测失败: {e}")
        sys.exit(1)

def run_test_detection(config):
    """运行测试检测"""
    print("运行测试检测...")
    
    try:
        # 测试基础检测器
        defender = DeepFaceDefender()
        
        # 创建测试图像
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 检测
        result = defender.detect_deepface_usage(test_image)
        
        print("\n测试结果:")
        print(f"是否为DeepFace: {result['is_deepface']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"攻击类型: {result['attack_type']}")
        print(f"详细信息: {result['details']}")
        
        # 显示统计信息
        stats = defender.get_detection_statistics()
        print(f"\n检测统计: {stats}")
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试检测失败: {e}")
        sys.exit(1)

def show_help():
    """显示帮助信息"""
    print("""
DeepFace检测和防御系统使用说明:

1. 视频检测模式:
   python main.py --mode video --input video.mp4 --output results.json

2. 实时检测模式:
   python main.py --mode realtime --camera 0

3. 测试模式:
   python main.py --mode test

4. 参数说明:
   --mode: 运行模式 (video/realtime/test)
   --input: 输入视频文件路径
   --output: 输出结果文件路径
   --threshold: 检测置信度阈值 (0.0-1.0)
   --camera: 摄像头ID (实时模式)

5. 配置文件:
   系统会自动创建 config.json 配置文件
   可以修改配置文件来调整检测参数

6. 输出文件:
   - results.json: 检测结果
   - anti_deepface.log: 日志文件
   - suspicious_frame_*.jpg: 可疑帧图像

7. 检测原理:
   - 图像质量分析
   - 统计特性检测
   - 频域特征分析
   - 噪声模式检测
   - 多维度综合评估

8. 安全建议:
   - 定期更新检测模型
   - 监控检测日志
   - 设置合适的阈值
   - 结合人工审核
""")

if __name__ == "__main__":
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        show_help()
    else:
        main()
