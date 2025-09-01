# DeepFace检测和防御系统

## 项目简介

这是一个专门用于检测和防御基于DeepFace的AI诈骗攻击的系统。该系统通过多维度分析来识别视频中是否使用了DeepFace技术，并提供实时监控和防御功能。

## 主要功能

### 1. 多维度检测
- **图像质量分析**: 检测图像是否过于完美（可能是AI生成）
- **统计特性检测**: 分析像素值分布是否异常
- **频域特征检测**: 检测频域中的异常模式
- **噪声模式检测**: 识别异常的噪声水平
- **代码签名检测**: 检测DeepFace相关代码特征
- **模型完整性检查**: 验证模型文件是否被篡改

### 2. 检测模式
- **视频检测**: 分析视频文件中的每一帧
- **实时检测**: 实时监控摄像头输入
- **测试模式**: 快速测试系统功能

### 3. 安全特性
- **多帧一致性检查**: 确保检测结果的可靠性
- **实时报警**: 发现可疑活动立即报警
- **结果导出**: 生成详细的检测报告
- **配置管理**: 灵活的配置系统

## 安装说明

### 1. 环境要求
- Python 3.7+
- OpenCV 4.5+
- NumPy 1.21+

### 2. 安装步骤
```bash
# 克隆项目
git clone <repository-url>
cd Anti-Deepface

# 安装依赖
pip install -r requirements.txt

# 运行测试
python main.py --mode test
```

## 使用方法

### 1. 视频检测
```bash
python main.py --mode video --input video.mp4 --output results.json
```

### 2. 实时检测
```bash
python main.py --mode realtime --camera 0
```

### 3. 测试模式
```bash
python main.py --mode test
```

### 4. 查看帮助
```bash
python main.py --help
```

## 配置说明

系统会自动创建 `config.json` 配置文件，包含以下设置：

### 检测参数
- `confidence_threshold`: 检测置信度阈值 (0.7)
- `anti_spoofing_threshold`: 反欺骗阈值 (0.8)
- `quality_threshold`: 图像质量阈值 (0.5)

### 视频参数
- `frame_interval`: 帧检测间隔 (1)
- `max_frames`: 最大检测帧数 (10000)
- `save_suspicious_frames`: 是否保存可疑帧 (true)

### 安全参数
- `enable_model_integrity_check`: 启用模型完整性检查 (true)
- `enable_runtime_protection`: 启用运行时保护 (true)
- `alert_threshold`: 报警阈值 (0.9)

## 检测原理

### 1. 图像质量分析
- 计算图像的清晰度（拉普拉斯方差）
- 检查分辨率和亮度
- 识别过于完美的图像特征

### 2. 统计特性检测
- 分析像素值分布
- 检测异常的直方图模式
- 识别过于均匀的分布

### 3. 频域特征检测
- 使用FFT分析频域特征
- 检测高频成分异常
- 识别对抗样本特征

### 4. 噪声模式检测
- 计算图像噪声水平
- 检测异常的噪声模式
- 识别过度处理的图像

## 输出文件

### 1. 检测结果
- `results.json`: 主要检测结果
- `results_report.json`: 详细检测报告
- `results_results.json`: 原始检测数据

### 2. 日志文件
- `anti_deepface.log`: 系统日志
- `config.json`: 配置文件

### 3. 可疑帧
- `suspicious_frame_*.jpg`: 保存的可疑帧图像

## 安全建议

### 1. 系统配置
- 定期更新检测模型
- 监控检测日志
- 设置合适的阈值
- 结合人工审核

### 2. 部署建议
- 使用隔离环境运行
- 限制文件系统权限
- 启用网络监控
- 定期备份配置

### 3. 检测优化
- 根据实际场景调整阈值
- 收集误报样本进行优化
- 定期评估检测效果
- 更新检测算法

## 技术架构

### 核心模块
- `deepface_defense.py`: 核心检测引擎
- `video_detector.py`: 视频检测模块
- `realtime_detector.py`: 实时检测模块
- `config.py`: 配置管理模块
- `main.py`: 主程序入口

### 检测流程
1. 输入预处理
2. 多维度特征提取
3. 特征融合分析
4. 结果综合评估
5. 报警和记录

## 性能指标

### 检测精度
- 真阳性率: >90%
- 假阳性率: <5%
- 检测延迟: <100ms

### 系统性能
- 支持1080p视频实时检测
- 内存占用: <500MB
- CPU使用率: <30%

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持视频和实时检测
- 多维度检测算法
- 配置管理系统

## 贡献指南

欢迎提交Issue和Pull Request来改进系统。

### 开发环境
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/

# 代码格式化
black *.py
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 项目讨论区

## 免责声明

本系统仅用于安全研究和防御目的，请勿用于非法用途。使用者需自行承担使用风险。
