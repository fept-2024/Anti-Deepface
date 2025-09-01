#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFace检测和防御系统 - 精简版
用于识别和防御基于DeepFace的AI诈骗攻击
"""

import os
import json
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anti_deepface.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DetectionLevel(Enum):
    """检测级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AttackType(Enum):
    """攻击类型枚举"""
    DEEPFAKE = "deepfake"
    SPOOFING = "spoofing"
    REPLAY = "replay"
    ADVERSARIAL = "adversarial"

@dataclass
class DetectionResult:
    """检测结果数据类"""
    is_deepface: bool
    confidence: float
    attack_type: Optional[AttackType]
    detection_level: DetectionLevel
    details: Dict[str, Any]
    timestamp: float
    frame_id: Optional[int] = None

@dataclass
class SecurityConfig:
    """安全配置数据类"""
    confidence_threshold: float = 0.7
    alert_threshold: float = 0.9
    time_window_seconds: int = 30
    max_frames_per_window: int = 100

class DeepFaceDetector:
    """DeepFace检测器主类"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.detection_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=100)
        logger.info("DeepFace检测器初始化完成")
    
    def detect_frame(self, frame: np.ndarray, frame_id: int = None) -> DetectionResult:
        """检测单帧图像"""
        start_time = time.time()
        
        # 多维度检测
        detection_results = {}
        
        # 1. 图像质量检测
        detection_results['image_quality'] = self._check_image_quality(frame)
        
        # 2. 统计特性检测
        detection_results['statistical'] = self._check_statistical_properties(frame)
        
        # 3. 频域特征检测
        detection_results['frequency'] = self._check_frequency_domain(frame)
        
        # 4. 噪声模式检测
        detection_results['noise_pattern'] = self._check_noise_patterns(frame)
        
        # 综合评估
        final_result = self._aggregate_results(detection_results, frame_id)
        final_result.timestamp = start_time
        
        # 记录检测历史
        self.detection_history.append(final_result)
        
        # 检查是否需要报警
        if final_result.is_deepface and final_result.confidence > self.config.alert_threshold:
            self._trigger_alert(final_result)
        
        return final_result
    
    def _check_image_quality(self, frame: np.ndarray) -> Dict[str, Any]:
        """检查图像质量"""
        if frame is None:
            return {'confidence': 0.0, 'details': {'method': 'image_quality'}}
        
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
        return {
            'confidence': min(quality_score, 1.0),
            'details': {
                'method': 'image_quality',
                'laplacian_var': laplacian_var,
                'resolution_score': resolution_score,
                'brightness_score': brightness_score
            }
        }
    
    def _check_statistical_properties(self, frame: np.ndarray) -> Dict[str, Any]:
        """检查统计特性"""
        if frame is None:
            return {'confidence': 0.0, 'details': {'method': 'statistical'}}
        
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
        confidence = min(anomalies / len(channels), 1.0)
        return {
            'confidence': confidence,
            'details': {
                'method': 'statistical',
                'anomalies': anomalies,
                'total_channels': len(channels)
            }
        }
    
    def _check_frequency_domain(self, frame: np.ndarray) -> Dict[str, Any]:
        """检查频域特征"""
        if frame is None:
            return {'confidence': 0.0, 'details': {'method': 'frequency'}}
        
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
        confidence = min(high_freq_mean / 10, 1.0)
        
        return {
            'confidence': confidence,
            'details': {
                'method': 'frequency',
                'high_freq_mean': high_freq_mean
            }
        }
    
    def _check_noise_patterns(self, frame: np.ndarray) -> Dict[str, Any]:
        """检查噪声模式"""
        if frame is None:
            return {'confidence': 0.0, 'details': {'method': 'noise_pattern'}}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯滤波计算噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred)
        noise_level = np.mean(noise)
        
        # 噪声水平过低可能表示图像被过度处理
        if noise_level < 5:
            confidence = 0.8
        elif noise_level > 50:
            confidence = 0.6
        else:
            confidence = 0.2
        
        return {
            'confidence': confidence,
            'details': {
                'method': 'noise_pattern',
                'noise_level': noise_level
            }
        }
    
    def _aggregate_results(self, results: Dict[str, Dict], frame_id: int) -> DetectionResult:
        """聚合多个模块的检测结果"""
        # 权重配置
        weights = {
            'image_quality': 0.3,
            'statistical': 0.25,
            'frequency': 0.2,
            'noise_pattern': 0.25
        }
        
        total_confidence = 0.0
        total_weight = 0.0
        details = {}
        
        for module_name, result in results.items():
            weight = weights.get(module_name, 0.1)
            confidence = result.get('confidence', 0.0)
            
            total_confidence += confidence * weight
            total_weight += weight
            details[module_name] = result.get('details', {})
        
        if total_weight > 0:
            final_confidence = total_confidence / total_weight
        else:
            final_confidence = 0.0
        
        # 确定检测级别
        if final_confidence >= 0.9:
            level = DetectionLevel.CRITICAL
        elif final_confidence >= 0.7:
            level = DetectionLevel.HIGH
        elif final_confidence >= 0.5:
            level = DetectionLevel.MEDIUM
        else:
            level = DetectionLevel.LOW
        
        # 判断是否为DeepFace
        is_deepface = final_confidence >= self.config.confidence_threshold
        
        # 确定攻击类型
        attack_type = self._determine_attack_type(details)
        
        return DetectionResult(
            is_deepface=is_deepface,
            confidence=final_confidence,
            attack_type=attack_type,
            detection_level=level,
            details=details,
            frame_id=frame_id
        )
    
    def _determine_attack_type(self, details: Dict[str, Dict]) -> Optional[AttackType]:
        """确定攻击类型"""
        # 基于检测详情判断攻击类型
        if details.get('frequency', {}).get('high_freq_mean', 0) > 8:
            return AttackType.ADVERSARIAL
        elif details.get('statistical', {}).get('anomalies', 0) >= 2:
            return AttackType.DEEPFAKE
        elif details.get('noise_pattern', {}).get('noise_level', 0) < 5:
            return AttackType.SPOOFING
        else:
            return AttackType.DEEPFAKE
    
    def _trigger_alert(self, result: DetectionResult):
        """触发安全报警"""
        alert = {
            'timestamp': time.time(),
            'frame_id': result.frame_id,
            'confidence': result.confidence,
            'attack_type': result.attack_type.value if result.attack_type else None,
            'detection_level': result.detection_level.value
        }
        self.alert_history.append(alert)
        logger.warning(f"🚨 安全报警: 检测到DeepFace使用！置信度: {result.confidence:.3f}")
    
    def detect_video(self, video_path: str) -> List[DetectionResult]:
        """检测视频中的DeepFace使用"""
        results = []
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                result = self.detect_frame(frame, frame_count)
                results.append(result)
                
                # 实时显示检测结果
                if result.is_deepface:
                    logger.warning(f"帧 {frame_count}: 检测到DeepFace使用，置信度: {result.confidence:.3f}")
                
                # 每100帧输出一次进度
                if frame_count % 100 == 0:
                    logger.info(f"已处理 {frame_count} 帧")
        
        finally:
            cap.release()
        
        return results
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        if not self.detection_history:
            return {}
        
        total_detections = len(self.detection_history)
        deepface_detections = sum(1 for r in self.detection_history if r.is_deepface)
        
        confidence_scores = [r.confidence for r in self.detection_history]
        
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
            'config': self.config.__dict__,
            'statistics': self.get_detection_statistics(),
            'detections': [
                {
                    'frame_id': r.frame_id,
                    'timestamp': r.timestamp,
                    'is_deepface': r.is_deepface,
                    'confidence': r.confidence,
                    'attack_type': r.attack_type.value if r.attack_type else None,
                    'detection_level': r.detection_level.value,
                    'details': r.details
                }
                for r in self.detection_history
            ],
            'alerts': list(self.alert_history)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"检测结果已导出到: {output_path}")

if __name__ == "__main__":
    # 测试代码
    detector = DeepFaceDetector()
    print("DeepFace检测和防御系统已启动")
    print("请使用 detect_video() 方法检测视频文件")
