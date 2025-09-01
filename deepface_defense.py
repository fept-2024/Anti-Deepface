#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFace防御系统核心模块
"""

import cv2
import numpy as np
import hashlib
import os
import time
import json
from typing import Dict, Any
from collections import deque

class DeepFaceDefender:
    """DeepFace防御器主类"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=1000)
        self.alert_threshold = 0.7
        self.model_hashes = {}
        
    def detect_deepface_usage(self, frame: np.ndarray) -> Dict[str, Any]:
        """检测DeepFace使用"""
        results = {
            'is_deepface': False,
            'confidence': 0.0,
            'attack_type': None,
            'details': {}
        }
        
        # 多维度检测
        confidence_scores = []
        
        # 1. 图像质量检测
        quality_score = self._check_image_quality(frame)
        confidence_scores.append(quality_score)
        
        # 2. 统计特性检测
        stats_score = self._check_statistical_properties(frame)
        confidence_scores.append(stats_score)
        
        # 3. 频域特征检测
        freq_score = self._check_frequency_domain(frame)
        confidence_scores.append(freq_score)
        
        # 4. 噪声模式检测
        noise_score = self._check_noise_patterns(frame)
        confidence_scores.append(noise_score)
        
        # 综合评分
        final_confidence = np.mean(confidence_scores)
        results['confidence'] = final_confidence
        results['is_deepface'] = final_confidence > self.alert_threshold
        
        # 确定攻击类型
        if results['is_deepface']:
            results['attack_type'] = self._determine_attack_type(confidence_scores)
        
        results['details'] = {
            'quality_score': quality_score,
            'stats_score': stats_score,
            'freq_score': freq_score,
            'noise_score': noise_score
        }
        
        # 记录检测历史
        self.detection_history.append({
            'timestamp': time.time(),
            'confidence': final_confidence,
            'is_deepface': results['is_deepface']
        })
        
        return results
    
    def _check_image_quality(self, frame: np.ndarray) -> float:
        """检查图像质量"""
        if frame is None:
            return 0.0
        
        # 计算清晰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 检查分辨率
        height, width = frame.shape[:2]
        resolution_score = min(width * height / (1920 * 1080), 1.0)
        
        # 检查亮度
        brightness = np.mean(gray)
        brightness_score = 1.0 if 50 < brightness < 200 else 0.5
        
        # 综合评分
        quality_score = (laplacian_var / 1000) * 0.4 + resolution_score * 0.3 + brightness_score * 0.3
        return min(quality_score, 1.0)
    
    def _check_statistical_properties(self, frame: np.ndarray) -> float:
        """检查统计特性"""
        if frame is None:
            return 0.0
        
        channels = cv2.split(frame)
        anomalies = 0
        
        for channel in channels:
            # 检查像素值分布
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # 检查分布是否过于均匀（可能是生成的）
            hist_std = np.std(hist)
            hist_mean = np.mean(hist)
            
            if hist_std < hist_mean * 0.1:  # 分布过于均匀
                anomalies += 1
        
        # 异常越多，越可能是DeepFace
        return min(anomalies / len(channels), 1.0)
    
    def _check_frequency_domain(self, frame: np.ndarray) -> float:
        """检查频域特征"""
        if frame is None:
            return 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # 检查高频成分
        height, width = magnitude_spectrum.shape
        center_y, center_x = height // 2, width // 2
        
        # 计算高频区域
        high_freq_region = magnitude_spectrum[
            center_y-height//4:center_y+height//4,
            center_x-width//4:center_x+width//4
        ]
        
        # 高频成分异常可能表示对抗样本
        high_freq_mean = np.mean(high_freq_region)
        return min(high_freq_mean / 10, 1.0)
    
    def _check_noise_patterns(self, frame: np.ndarray) -> float:
        """检查噪声模式"""
        if frame is None:
            return 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯滤波计算噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred)
        noise_level = np.mean(noise)
        
        # 噪声水平过低可能表示图像被过度处理
        if noise_level < 5:
            return 0.8
        elif noise_level > 50:
            return 0.6
        else:
            return 0.2
    
    def _determine_attack_type(self, confidence_scores: list) -> str:
        """确定攻击类型"""
        if confidence_scores[0] > 0.8:  # 图像质量异常
            return "deepfake"
        elif confidence_scores[1] > 0.8:  # 统计特性异常
            return "adversarial"
        elif confidence_scores[2] > 0.8:  # 频域异常
            return "perturbation"
        else:
            return "unknown"
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        if not self.detection_history:
            return {}
        
        total_detections = len(self.detection_history)
        deepface_detections = sum(1 for r in self.detection_history if r['is_deepface'])
        
        confidence_scores = [r['confidence'] for r in self.detection_history]
        
        return {
            'total_frames': total_detections,
            'deepface_detections': deepface_detections,
            'detection_rate': deepface_detections / total_detections if total_detections > 0 else 0,
            'avg_confidence': np.mean(confidence_scores),
            'max_confidence': np.max(confidence_scores),
            'min_confidence': np.min(confidence_scores)
        }
    
    def export_results(self, output_path: str):
        """导出检测结果"""
        results = {
            'statistics': self.get_detection_statistics(),
            'detections': list(self.detection_history)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"检测结果已导出到: {output_path}")

def test_defender():
    """测试防御器"""
    defender = DeepFaceDefender()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 检测
    result = defender.detect_deepface_usage(test_image)
    
    print("DeepFace防御系统测试结果:")
    print(f"是否为DeepFace: {result['is_deepface']}")
    print(f"置信度: {result['confidence']:.3f}")
    print(f"攻击类型: {result['attack_type']}")
    print(f"详细信息: {result['details']}")

if __name__ == "__main__":
    test_defender()
