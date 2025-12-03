# 猪发病率预测系统

## 项目概述

本项目基于神经网络算法，构建了一个猪群发病率预测系统，能够根据多种环境、饲养管理、生物安全等因素预测猪群的发病率风险，并通过FastAPI封装为HTTP接口，方便集成到各类养殖管理系统中。

## 项目结构

```
├── pig_disease_training_data.csv   # 训练数据文件
├── train_model.py                  # 神经网络模型训练代码
├── api.py                          # FastAPI接口封装代码
├── models/                         # 保存的模型和标量器
│   ├── pig_disease_model.pth       # 训练好的PyTorch模型
│   └── scaler.pkl                  # 数据标准化标量器
├── training_loss.png               # 训练过程损失曲线
├── README.md                       # 项目说明文档
├── 变量设计说明.md                  # 特征变量设计说明
├── 数据说明文档.md                  # 训练数据详细说明
└── 验证CSV文件.py                   # 数据验证脚本
```

## 技术栈

- **开发语言**: Python 3.8+
- **机器学习框架**: PyTorch
- **Web框架**: FastAPI
- **数据处理**: Pandas, NumPy, Scikit-learn
- **API服务**: Uvicorn

## 安装与配置

### 1. 环境准备

```bash
# 创建并激活虚拟环境
python -m venv .venv
# Windows激活
.venv\Scripts\activate
# Linux/Mac激活
# source .venv/bin/activate

# 安装依赖包
pip install torch fastapi uvicorn scikit-learn matplotlib joblib pandas numpy
```

### 2. 数据准备

项目包含了模拟的训练数据文件 `pig_disease_training_data.csv`，包含200条记录，涵盖22个特征变量和1个目标变量（发病率）。

数据详细说明请参考 `数据说明文档.md`。

## 模型训练

### 1. 运行训练脚本

```bash
python train_model.py
```

### 2. 训练过程说明

- 脚本会自动加载CSV数据，进行标准化处理
- 划分训练集（80%）和测试集（20%）
- 训练一个3层全连接神经网络（64-32-16节点）
- 使用Adam优化器，学习率为0.001，训练500个epoch
- 保存训练好的模型和标量器到 `models` 目录
- 生成训练损失曲线 `training_loss.png`

### 3. 模型评估结果

训练完成后，会输出模型在测试集上的评估指标：
- 均方根误差 (RMSE)
- 决定系数 (R²)

## API使用方法

### 1. 启动API服务

```bash
# 方法1：直接运行api.py
python api.py

# 方法2：使用uvicorn启动（推荐）
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. 访问API文档

API服务启动后，可以通过以下地址访问自动生成的API文档：
- **交互式API文档**: http://localhost:8000/docs
- **OpenAPI规范**: http://localhost:8000/openapi.json

## API接口文档

### 1. 健康检查接口

- **URL**: `/health`
- **方法**: `GET`
- **功能**: 检查API服务和模型是否正常运行
- **响应示例**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

### 2. 单条预测接口

- **URL**: `/predict`
- **方法**: `POST`
- **功能**: 根据输入的特征数据预测猪群发病率
- **请求体**:
  ```json
  {
    "temperature": 25.5,            # 环境温度(°C)
    "humidity": 65.2,              # 相对湿度(%)
    "ventilation_rate": 1200,      # 通风率
    "ammonia_level": 15.0,         # 氨水平
    "co2_level": 3000,             # 二氧化碳水平
    "pig_density": 0.8,            # 猪群密度
    "avg_feed_intake": 2.5,        # 平均采食量
    "water_quality": 4,            # 水质评分（5分制）
    "feed_quality": 4,             # 饲料质量评分（5分制）
    "bedding_condition": 3,        # 垫料条件（5分制）
    "vaccination_rate": 95.0,      # 疫苗接种率(%)
    "disinfection_frequency": 4,   # 消毒频率
    "quarantine_days": 14,         # 检疫天数
    "staff_biosecurity_score": 8,  # 员工生物安全评分
    "visitor_restriction_level": 5,# 访客限制等级
    "avg_age": 90,                 # 平均年龄(日龄)
    "breed_type": 2,               # 品种类型
    "health_score": 7,             # 健康评分
    "previous_incidence": 20,      # 前一周发病率(%)
    "mortality_rate_prev": 8,      # 前一周死亡率(%)
    "season": 2,                   # 季节(1-4)
    "is_rainy_season": 0           # 是否雨季(0/1)
  }
  ```
- **响应示例**:
  ```json
  {
    "predicted_incidence": 1.9,       # 预测的发病率(%)
    "confidence": 0.85,              # 预测置信度
    "risk_level": "中等风险",        # 风险等级
    "recommendations": [             # 防控建议
      "加强环境消毒频率",
      "检查饲料和饮水质量",
      "降低养殖密度",
      "增加兽医巡查频次"
    ]
  }
  ```

### 3. 批量预测接口

- **URL**: `/batch_predict`
- **方法**: `POST`
- **功能**: 批量预测多条数据的猪群发病率
- **请求体**:
  ```json
  {
    "data": [
      {
        "temperature": 25.5,
        "humidity": 65.2,
        "ventilation_rate": 1200,
        "ammonia_level": 15.0,
        "co2_level": 3000,
        "pig_density": 0.8,
        "avg_feed_intake": 2.5,
        "water_quality": 4,
        "feed_quality": 4,
        "bedding_condition": 3,
        "vaccination_rate": 95.0,
        "disinfection_frequency": 4,
        "quarantine_days": 14,
        "staff_biosecurity_score": 8,
        "visitor_restriction_level": 5,
        "avg_age": 90,
        "breed_type": 2,
        "health_score": 7,
        "previous_incidence": 20,
        "mortality_rate_prev": 8,
        "season": 2,
        "is_rainy_season": 0
      },
      {
        "temperature": 26.0,
        "humidity": 70.0,
        "ventilation_rate": 1100,
        "ammonia_level": 20.0,
        "co2_level": 3500,
        "pig_density": 0.9,
        "avg_feed_intake": 2.3,
        "water_quality": 3,
        "feed_quality": 3,
        "bedding_condition": 3,
        "vaccination_rate": 92.0,
        "disinfection_frequency": 3,
        "quarantine_days": 10,
        "staff_biosecurity_score": 7,
        "visitor_restriction_level": 4,
        "avg_age": 85,
        "breed_type": 3,
        "health_score": 6,
        "previous_incidence": 25,
        "mortality_rate_prev": 10,
        "season": 3,
        "is_rainy_season": 1
      }
    ]
  }
  ```
- **响应示例**:
  ```json
  {
    "predictions": [
      {
        "predicted_incidence": 1.9,
        "confidence": 0.85,
        "risk_level": "中等风险",
        "recommendations": ["建议1", "建议2", ...]
      },
      {
        "predicted_incidence": 3.2,
        "confidence": 0.80,
        "risk_level": "高风险",
        "recommendations": ["建议1", "建议2", ...]
      }
    ]
  }
  ```

### 4. 模型信息接口

- **URL**: `/model/info`
- **方法**: `GET`
- **功能**: 获取模型的基本信息
- **响应示例**:
  ```json
  {
    "model_type": "神经网络",
    "input_features": 22,
    "hidden_layers": [64, 32, 16],
    "output": "猪群发病率预测",
    "framework": "PyTorch",
    "version": "1.0"
  }
  ```

## 示例请求

### PowerShell示例

```powershell
# 健康检查
Invoke-WebRequest -Uri "http://localhost:8000/health" -Method Get | Select-Object -Expand Content

# 单条预测
Invoke-WebRequest -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{"temperature": 25.5, "humidity": 65.2, "ventilation_rate": 1200, "ammonia_level": 15.0, "co2_level": 3000, "pig_density": 0.8, "avg_feed_intake": 2.5, "water_quality": 4, "feed_quality": 4, "bedding_condition": 3, "vaccination_rate": 95.0, "disinfection_frequency": 4, "quarantine_days": 14, "staff_biosecurity_score": 8, "visitor_restriction_level": 5, "avg_age": 90, "breed_type": 2, "health_score": 7, "previous_incidence": 20, "mortality_rate_prev": 8, "season": 2, "is_rainy_season": 0}' | Select-Object -Expand Content
```

### Python示例

```python
import requests
import json

# 单条预测
url = "http://localhost:8000/predict"
headers = {"Content-Type": "application/json"}

data = {
    "temperature": 25.5,
    "humidity": 65.2,
    "ventilation_rate": 1200,
    "ammonia_level": 15.0,
    "co2_level": 3000,
    "pig_density": 0.8,
    "avg_feed_intake": 2.5,
    "water_quality": 4,
    "feed_quality": 4,
    "bedding_condition": 3,
    "vaccination_rate": 95.0,
    "disinfection_frequency": 4,
    "quarantine_days": 14,
    "staff_biosecurity_score": 8,
    "visitor_restriction_level": 5,
    "avg_age": 90,
    "breed_type": 2,
    "health_score": 7,
    "previous_incidence": 20,
    "mortality_rate_prev": 8,
    "season": 2,
    "is_rainy_season": 0
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
```

## 风险等级说明

| 风险等级 | 发病率范围 | 置信度 | 应对策略 |
|---------|-----------|--------|--------|
| 低风险  | < 1.0%    | 0.95   | 维持当前管理水平 |
| 中等风险 | 1.0% - 3.0% | 0.85 | 加强监测和预防 |
| 高风险  | 3.0% - 5.0% | 0.80 | 采取紧急防控措施 |
| 极高风险 | > 5.0%    | 0.75   | 启动应急预案 |

## 注意事项

1. **数据质量**：请确保输入数据在合理范围内，API会进行参数验证
2. **模型更新**：随着新数据的积累，建议定期重新训练模型以提高预测准确性
3. **API部署**：生产环境部署时，建议使用Gunicorn+Uvicorn的组合以提高性能
4. **并发处理**：API支持批量预测功能，可以减少多次请求的开销
5. **监控告警**：建议设置系统监控，当预测风险等级为高风险或极高风险时触发告警

## 扩展与优化建议

1. **模型优化**：可以尝试其他深度学习架构（如LSTM、CNN）或集成学习方法
2. **实时数据**：接入物联网设备，获取实时环境和猪群数据
3. **多模型融合**：结合传统统计模型和深度学习模型，提高预测稳定性
4. **特征工程**：进一步挖掘和构建更有价值的特征变量
5. **可解释性增强**：添加模型解释功能，分析各因素对发病率的影响权重

## 许可证

本项目仅供学术研究和技术演示使用。
