<h1 align="center">咸宁方言语音分类模型</h1>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</div>

## 目录
- [1. 更新日志](#1-更新日志)
- [2. 项目简介](#2-项目简介)
- [3. 环境配置](#3-环境配置)
- [4. 快速开始](#4-快速开始)
- [5. 项目结构](#5-项目结构)
- [6. 模型训练](#6-模型训练)
- [7. 模型评估](#7-模型评估)
- [8. API接口](#8-api接口)
- [9. 性能指标](#9-性能指标)
- [10. 常见问题](#10-常见问题)
- [11. 贡献指南](#11-贡献指南)
- [12. 联系我们](#12-联系我们)

## 1. 更新日志
### 1.0 (2025-06-04)
- 初始版本发布
- 包含基础代码框架
- 提供环境配置文件
- 发布预训练模型

### 1.1 (2025-06-11)
- 完善README文档
- 优化训练脚本
- 修复已知问题

### 1.2 (2025-06-24)
- 新增了基础版本（CNN）的模型训练代码作为前置学习代码
## 2. 项目简介
本项目是基于wav2vec2.0框架训练的语音分类模型，专门针对中国湖北省咸宁地区六县市的方言进行分类识别。

### 主要特性
✅ 支持咸宁6县市方言分类  
✅ 基于Transformer的高效语音特征提取  
✅ 提供预训练模型和训练代码  
✅ 易于部署的REST API接口  

> 📝 **注意**：训练数据集因隐私原因不予公开，但提供示例数据供测试使用。

## 3. 环境配置
### 基础要求
| 组件 | 版本要求 |
|------|----------|
| Python | 3.8+ |
| PyTorch | 1.12+ |
| CUDA | 11.3+ (GPU用户) |
### 核心依赖
torch>=1.12.0
transformers>=4.18.0
librosa>=0.9.2
numpy>=1.21.5
pandas>=1.4.0

### 安装步骤
```bash
# 创建conda环境
conda create -n xianning_asr python=3.8
conda activate xianning_asr

# 安装依赖
pip install -r requirements.txt

```

## 4.后续待更新
