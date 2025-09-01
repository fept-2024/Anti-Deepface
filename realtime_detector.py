#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时DeepFace检测器
"""

import cv2
import numpy as np
import time
from deepface_defense import DeepFaceDefender

class RealtimeDetector:
    """实时检测器"""
    
    def __init__(self, camera_id=0):
        self.defender = DeepFaceDefender()
        self.camera_id = camera_id
        self.is_running = False
        
    def start_detection(self):
        """开始检测"""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        self.is_running = True
        print("开始实时检测，按 'q' 退出")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测
            result = self.defender.detect_deepface_usage(frame)
            
            # 显示结果
            self._draw_result(frame, result)
            
            # 显示画面
            cv2.imshow('DeepFace检测', frame)
            
            # 检查按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_result(self, frame, result):
        """在画面上绘制结果"""
        text = f"DeepFace: {'是' if result['is_deepface'] else '否'}"
        color = (0, 0, 255) if result['is_deepface'] else (0, 255, 0)
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"置信度: {result['confidence']:.3f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    detector = RealtimeDetector()
    detector.start_detection()

if __name__ == "__main__":
    main()
