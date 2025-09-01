#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频DeepFace检测器
用于检测视频中的DeepFace使用情况
"""

import cv2
import numpy as np
import os
import time
import json
from typing import List, Dict, Any
from deepface_defense import DeepFaceDefender

class VideoDeepFaceDetector:
    """视频DeepFace检测器"""
    
    def __init__(self):
        self.defender = DeepFaceDefender()
        self.results = []
        self.frame_interval = 1  # 每帧都检测，可以设置为更大的值来跳过帧
        
    def detect_video(self, video_path: str, output_path: str = None) -> Dict[str, Any]:
        """检测视频中的DeepFace使用"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        print(f"开始检测视频: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 每frame_interval帧检测一次
                if frame_count % self.frame_interval == 0:
                    # 检测当前帧
                    result = self.defender.detect_deepface_usage(frame)
                    result['frame_number'] = frame_count
                    result['timestamp'] = frame_count / fps
                    self.results.append(result)
                    
                    # 实时显示检测结果
                    if result['is_deepface']:
                        print(f"🚨 帧 {frame_count}: 检测到DeepFace使用！置信度: {result['confidence']:.3f}")
                        print(f"   攻击类型: {result['attack_type']}")
                
                # 显示进度
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            cap.release()
        
        # 生成检测报告
        report = self._generate_report(video_path, total_frames, fps)
        
        # 导出结果
        if output_path:
            self._export_results(output_path, report)
        
        return report
    
    def _generate_report(self, video_path: str, total_frames: int, fps: float) -> Dict[str, Any]:
        """生成检测报告"""
        if not self.results:
            return {
                'video_path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'detection_summary': '未检测到DeepFace使用',
                'statistics': {}
            }
        
        # 统计信息
        detected_frames = [r for r in self.results if r['is_deepface']]
        detection_rate = len(detected_frames) / len(self.results)
        
        # 置信度统计
        confidence_scores = [r['confidence'] for r in self.results]
        
        # 攻击类型统计
        attack_types = {}
        for result in detected_frames:
            attack_type = result['attack_type']
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
        
        # 时间线分析
        timeline = []
        for result in self.results:
            if result['is_deepface']:
                timeline.append({
                    'timestamp': result['timestamp'],
                    'frame': result['frame_number'],
                    'confidence': result['confidence'],
                    'attack_type': result['attack_type']
                })
        
        # 风险评估
        risk_level = self._assess_risk(detection_rate, confidence_scores)
        
        report = {
            'video_path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'analyzed_frames': len(self.results),
            'detection_summary': f"检测到 {len(detected_frames)} 帧使用DeepFace",
            'risk_level': risk_level,
            'statistics': {
                'detection_rate': detection_rate,
                'avg_confidence': np.mean(confidence_scores),
                'max_confidence': np.max(confidence_scores),
                'min_confidence': np.min(confidence_scores),
                'attack_types': attack_types
            },
            'timeline': timeline,
            'recommendations': self._generate_recommendations(detection_rate, risk_level)
        }
        
        return report
    
    def _assess_risk(self, detection_rate: float, confidence_scores: List[float]) -> str:
        """评估风险等级"""
        avg_confidence = np.mean(confidence_scores)
        max_confidence = np.max(confidence_scores)
        
        if detection_rate > 0.5 or max_confidence > 0.9:
            return "高风险"
        elif detection_rate > 0.2 or avg_confidence > 0.7:
            return "中风险"
        elif detection_rate > 0.05 or avg_confidence > 0.5:
            return "低风险"
        else:
            return "安全"
    
    def _generate_recommendations(self, detection_rate: float, risk_level: str) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if risk_level == "高风险":
            recommendations.extend([
                "立即停止使用该视频",
                "进行人工审核确认",
                "加强反欺骗检测机制",
                "考虑使用多模态验证"
            ])
        elif risk_level == "中风险":
            recommendations.extend([
                "谨慎使用该视频",
                "增加额外的验证步骤",
                "监控后续使用情况"
            ])
        elif risk_level == "低风险":
            recommendations.extend([
                "可以正常使用，但保持警惕",
                "定期进行安全检测"
            ])
        else:
            recommendations.append("视频安全，可以正常使用")
        
        return recommendations
    
    def _export_results(self, output_path: str, report: Dict[str, Any]):
        """导出检测结果"""
        # 保存详细报告
        report_path = output_path.replace('.json', '_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存原始检测结果
        results_path = output_path.replace('.json', '_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"检测报告已保存到: {report_path}")
        print(f"详细结果已保存到: {results_path}")
    
    def print_summary(self, report: Dict[str, Any]):
        """打印检测摘要"""
        print("\n" + "="*50)
        print("DeepFace检测报告")
        print("="*50)
        print(f"视频文件: {report['video_path']}")
        print(f"总帧数: {report['total_frames']}")
        print(f"分析帧数: {report['analyzed_frames']}")
        print(f"检测摘要: {report['detection_summary']}")
        print(f"风险等级: {report['risk_level']}")
        print(f"检测率: {report['statistics']['detection_rate']:.2%}")
        print(f"平均置信度: {report['statistics']['avg_confidence']:.3f}")
        print(f"最高置信度: {report['statistics']['max_confidence']:.3f}")
        
        if report['statistics']['attack_types']:
            print("攻击类型分布:")
            for attack_type, count in report['statistics']['attack_types'].items():
                print(f"  {attack_type}: {count} 帧")
        
        print("\n建议:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        print("="*50)

def main():
    """主函数"""
    detector = VideoDeepFaceDetector()
    
    # 测试视频路径（需要用户提供）
    video_path = input("请输入要检测的视频文件路径: ").strip()
    
    if not video_path:
        print("未提供视频路径，使用默认测试")
        return
    
    try:
        # 检测视频
        report = detector.detect_video(video_path, "detection_results.json")
        
        # 打印摘要
        detector.print_summary(report)
        
    except Exception as e:
        print(f"检测过程中出现错误: {e}")

if __name__ == "__main__":
    main()
