#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFace检测系统配置文件
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DetectionConfig:
    """检测配置"""
    # 基础检测参数
    confidence_threshold: float = 0.7
    anti_spoofing_threshold: float = 0.8
    consistency_threshold: float = 0.6
    
    # 图像质量检测参数
    min_resolution: tuple = (100, 100)
    max_resolution: tuple = (4096, 4096)
    quality_threshold: float = 0.5
    
    # 频域检测参数
    fft_threshold: float = 8.0
    high_freq_weight: float = 0.3
    
    # 噪声检测参数
    noise_low_threshold: float = 5.0
    noise_high_threshold: float = 50.0
    
    # 统计特性检测参数
    histogram_uniformity_threshold: float = 0.1

@dataclass
class VideoConfig:
    """视频检测配置"""
    # 帧处理参数
    frame_interval: int = 1  # 每N帧检测一次
    max_frames: int = 10000  # 最大检测帧数
    
    # 输出参数
    save_suspicious_frames: bool = True
    output_format: str = "json"
    
    # 实时检测参数
    realtime_fps: int = 30
    realtime_resolution: tuple = (640, 480)

@dataclass
class SecurityConfig:
    """安全配置"""
    # 模型完整性检查
    enable_model_integrity_check: bool = True
    model_hash_file: str = "model_hashes.json"
    
    # 运行时保护
    enable_runtime_protection: bool = True
    monitor_imports: bool = True
    monitor_api_calls: bool = True
    
    # 报警设置
    alert_threshold: float = 0.9
    alert_cooldown: float = 5.0  # 秒
    enable_email_alert: bool = False
    enable_sms_alert: bool = False

@dataclass
class LoggingConfig:
    """日志配置"""
    log_level: str = "INFO"
    log_file: str = "anti_deepface.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.detection = DetectionConfig()
        self.video = VideoConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        
        # 加载配置文件
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        config_file = "config.json"
        if os.path.exists(config_file):
            try:
                import json
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 更新配置
                self._update_config(config_data)
                print(f"配置文件已加载: {config_file}")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
        else:
            # 创建默认配置文件
            self._save_default_config()
    
    def _update_config(self, config_data: Dict[str, Any]):
        """更新配置"""
        if 'detection' in config_data:
            for key, value in config_data['detection'].items():
                if hasattr(self.detection, key):
                    setattr(self.detection, key, value)
        
        if 'video' in config_data:
            for key, value in config_data['video'].items():
                if hasattr(self.video, key):
                    setattr(self.video, key, value)
        
        if 'security' in config_data:
            for key, value in config_data['security'].items():
                if hasattr(self.security, key):
                    setattr(self.security, key, value)
        
        if 'logging' in config_data:
            for key, value in config_data['logging'].items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
    
    def _save_default_config(self):
        """保存默认配置文件"""
        config_data = {
            'detection': {
                'confidence_threshold': self.detection.confidence_threshold,
                'anti_spoofing_threshold': self.detection.anti_spoofing_threshold,
                'consistency_threshold': self.detection.consistency_threshold,
                'min_resolution': self.detection.min_resolution,
                'max_resolution': self.detection.max_resolution,
                'quality_threshold': self.detection.quality_threshold,
                'fft_threshold': self.detection.fft_threshold,
                'high_freq_weight': self.detection.high_freq_weight,
                'noise_low_threshold': self.detection.noise_low_threshold,
                'noise_high_threshold': self.detection.noise_high_threshold,
                'histogram_uniformity_threshold': self.detection.histogram_uniformity_threshold
            },
            'video': {
                'frame_interval': self.video.frame_interval,
                'max_frames': self.video.max_frames,
                'save_suspicious_frames': self.video.save_suspicious_frames,
                'output_format': self.video.output_format,
                'realtime_fps': self.video.realtime_fps,
                'realtime_resolution': self.video.realtime_resolution
            },
            'security': {
                'enable_model_integrity_check': self.security.enable_model_integrity_check,
                'model_hash_file': self.security.model_hash_file,
                'enable_runtime_protection': self.security.enable_runtime_protection,
                'monitor_imports': self.security.monitor_imports,
                'monitor_api_calls': self.security.monitor_api_calls,
                'alert_threshold': self.security.alert_threshold,
                'alert_cooldown': self.security.alert_cooldown,
                'enable_email_alert': self.security.enable_email_alert,
                'enable_sms_alert': self.security.enable_sms_alert
            },
            'logging': {
                'log_level': self.logging.log_level,
                'log_file': self.logging.log_file,
                'log_format': self.logging.log_format,
                'max_log_size': self.logging.max_log_size,
                'backup_count': self.logging.backup_count
            }
        }
        
        try:
            import json
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            print("默认配置文件已创建: config.json")
        except Exception as e:
            print(f"创建配置文件失败: {e}")
    
    def save_config(self):
        """保存当前配置"""
        self._save_default_config()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'detection_threshold': self.detection.confidence_threshold,
            'video_frame_interval': self.video.frame_interval,
            'security_protection': self.security.enable_runtime_protection,
            'logging_level': self.logging.log_level
        }

# 全局配置实例
config = ConfigManager()

def get_config() -> ConfigManager:
    """获取配置管理器"""
    return config
