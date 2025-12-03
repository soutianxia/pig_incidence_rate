from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np
import joblib
import os
from typing import List
import uvicorn

# 导入之前定义的神经网络模型结构
from train_model import NeuralNetwork

# 创建FastAPI应用实例
app = FastAPI(
    title="猪发病率预测API",
    description="使用神经网络模型预测猪群发病率的RESTful API",
    version="1.0.0"
)

# 全局变量用于存储模型和标量器
model = None
scaler = None
model_loaded = False

# 定义请求数据模型（与CSV文件字段保持一致）
class PredictionRequest(BaseModel):
    # 环境因素
    temperature: float = Field(..., description="环境温度(°C)", ge=0, le=50)
    humidity: float = Field(..., description="相对湿度(%)", ge=0, le=100)
    ventilation_rate: int = Field(..., description="通风率", ge=0, le=5000)
    ammonia_level: float = Field(..., description="氨水平", ge=0, le=50)
    co2_level: int = Field(..., description="二氧化碳水平", ge=0, le=10000)
    
    # 饲养管理因素
    pig_density: float = Field(..., description="猪群密度", ge=0, le=10)
    avg_feed_intake: float = Field(..., description="平均采食量", ge=0, le=10)
    water_quality: int = Field(..., description="水质评分", ge=0, le=10)
    feed_quality: int = Field(..., description="饲料质量评分", ge=0, le=10)
    bedding_condition: int = Field(..., description="垫料条件", ge=0, le=10)
    
    # 生物安全因素
    vaccination_rate: float = Field(..., description="疫苗接种率(%)", ge=0, le=100)
    disinfection_frequency: int = Field(..., description="消毒频率", ge=0, le=10)
    quarantine_days: int = Field(..., description="检疫天数", ge=0, le=30)
    staff_biosecurity_score: int = Field(..., description="员工生物安全评分", ge=0, le=10)
    visitor_restriction_level: int = Field(..., description="访客限制等级", ge=0, le=10)
    
    # 猪群特征
    avg_age: int = Field(..., description="平均年龄(日龄)", ge=0, le=500)
    breed_type: int = Field(..., description="品种类型", ge=1, le=5)
    health_score: int = Field(..., description="健康评分", ge=0, le=10)
    
    # 历史数据
    previous_incidence: float = Field(..., description="前一周发病率(%)", ge=0, le=100)
    mortality_rate_prev: float = Field(..., description="前一周死亡率(%)", ge=0, le=100)
    
    # 季节性因素
    season: int = Field(..., description="季节(1-4)", ge=1, le=4)
    is_rainy_season: int = Field(..., description="是否雨季(0/1)", ge=0, le=1)

# 定义响应数据模型
class PredictionResponse(BaseModel):
    predicted_incidence: float = Field(..., description="预测的发病率(%)")
    confidence: float = Field(..., description="预测置信度")
    risk_level: str = Field(..., description="风险等级")
    recommendations: List[str] = Field(..., description="防控建议")

# 批量预测请求模型
class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest]

# 批量预测响应模型
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# 加载模型和标量器
def load_model_and_scaler(model_dir='models'):
    global model, scaler, model_loaded
    
    try:
        # 创建模型实例
        model = NeuralNetwork(input_size=22, hidden_sizes=[64, 32, 16], output_size=1)
        
        # 加载模型权重
        model_path = os.path.join(model_dir, 'pig_disease_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载标量器
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"标量器文件不存在: {scaler_path}")
        
        model_loaded = True
        print("模型和标量器加载成功!")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        model_loaded = False

# 获取风险等级和建议
def get_risk_assessment(predicted_value):
    if predicted_value < 1.0:
        risk_level = "低风险"
        confidence = 0.95
        recommendations = [
            "维持当前的饲养管理水平",
            "继续执行常规的生物安全措施",
            "定期监测猪群健康状况"
        ]
    elif predicted_value < 3.0:
        risk_level = "中等风险"
        confidence = 0.85
        recommendations = [
            "加强环境消毒频率",
            "检查饲料和饮水质量",
            "降低养殖密度",
            "增加兽医巡查频次"
        ]
    elif predicted_value < 5.0:
        risk_level = "高风险"
        confidence = 0.80
        recommendations = [
            "立即加强生物安全措施",
            "对猪群进行全面健康检查",
            "调整饲养密度和环境参数",
            "增加疫苗接种覆盖率",
            "准备应急预案"
        ]
    else:
        risk_level = "极高风险"
        confidence = 0.75
        recommendations = [
            "立即隔离病猪",
            "全面消毒养殖场",
            "紧急联系兽医进行诊断",
            "考虑部分猪群转移",
            "启动疫病应急响应程序"
        ]
    
    return risk_level, confidence, recommendations

# 健康检查端点
@app.get("/health")
async def health_check():
    if model_loaded:
        return {"status": "healthy", "model_loaded": True}
    else:
        # 尝试重新加载模型
        load_model_and_scaler()
        return {"status": "healthy", "model_loaded": model_loaded}
# 在predict函数中，标准化之前增加极端值检查
def check_extreme_values(features):
    """检测极端特征值并直接返回极高风险"""
    # 定义各特征的极端值阈值
    extreme_thresholds = {
        'temperature': (0, 42),  # 温度超过42℃或低于0℃为极端
        'humidity': (0, 90),     # 湿度超过90%或低于0%为极端
        'ammonia_level': (0, 15),# 氨水平超过15为极端
        'co2_level': (0, 4000),  # CO2超过4000为极端
        'pig_density': (0, 3.0), # 猪密度超过3.0为极端
    }
    
    # 单独处理采食量（只有过低才是极端）
    feed_intake_min = 1.5
    
    feature_names = [
        'temperature', 'humidity', 'ventilation_rate', 'ammonia_level', 'co2_level',
        'pig_density', 'avg_feed_intake', 'water_quality', 'feed_quality', 'bedding_condition',
        'vaccination_rate', 'disinfection_frequency', 'quarantine_days', 'staff_biosecurity_score',
        'visitor_restriction_level', 'avg_age', 'breed_type', 'health_score',
        'previous_incidence', 'mortality_rate_prev', 'season', 'is_rainy_season'
    ]
    
    for i, feature_name in enumerate(feature_names):
        if feature_name in extreme_thresholds:
            min_val, max_val = extreme_thresholds[feature_name]
            if features[0, i] < min_val or features[0, i] > max_val:
                return True, f"{feature_name}值({features[0, i]})超出正常范围，属于极端情况"
        # 单独处理采食量（只有过低才是极端）
        elif feature_name == 'avg_feed_intake':
            if features[0, i] < feed_intake_min:
                return True, f"{feature_name}值({features[0, i]})过低，属于极端情况"
    
    return False, None
# 单条预测端点
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):

    if not model_loaded:
        # 尝试加载模型
        load_model_and_scaler()
        if not model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载，请稍后再试")
    
    try:
        # 提取特征值并转换为numpy数组
        features = np.array([
            request.temperature,
            request.humidity,
            request.ventilation_rate,
            request.ammonia_level,
            request.co2_level,
            request.pig_density,
            request.avg_feed_intake,
            request.water_quality,
            request.feed_quality,
            request.bedding_condition,
            request.vaccination_rate,
            request.disinfection_frequency,
            request.quarantine_days,
            request.staff_biosecurity_score,
            request.visitor_restriction_level,
            request.avg_age,
            request.breed_type,
            request.health_score,
            request.previous_incidence,
            request.mortality_rate_prev,
            request.season,
            request.is_rainy_season
        ]).reshape(1, -1)
        
        # 检查是否存在极端值
        is_extreme, reason = check_extreme_values(features)
        if is_extreme:
            # 对于极端值，直接返回极高风险
            return PredictionResponse(
                predicted_incidence=6.0,  # 固定返回超过5.0%的值
                confidence=0.85,
                risk_level="极高风险",
                recommendations=[
                    f"检测到极端值: {reason}",
                    "立即采取紧急措施",
                    "将猪群转移到适宜环境",
                    "紧急联系兽医",
                    "启动应急预案"
                ]
            )
        
        # 使用标量器进行数据标准化
        features_scaled = scaler.transform(features)
        
        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(features_scaled)
        
        # 进行预测
        with torch.no_grad():
            prediction = model(features_tensor).item()
        
        # 确保预测值在合理范围内
        predicted_incidence = max(0.0, min(prediction, 100.0))
        
        # 获取风险评估和建议
        risk_level, confidence, recommendations = get_risk_assessment(predicted_incidence)
        
        return PredictionResponse(
            predicted_incidence=round(predicted_incidence, 2),
            confidence=confidence,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预测过程中出错: {str(e)}")

# 批量预测端点
@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    if not model_loaded:
        # 尝试加载模型
        load_model_and_scaler()
        if not model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载，请稍后再试")
    
    try:
        predictions = []
        
        for item in request.data:
            # 提取特征值
            features = np.array([
                item.temperature,
                item.humidity,
                item.ventilation_rate,
                item.ammonia_level,
                item.co2_level,
                item.pig_density,
                item.avg_feed_intake,
                item.water_quality,
                item.feed_quality,
                item.bedding_condition,
                item.vaccination_rate,
                item.disinfection_frequency,
                item.quarantine_days,
                item.staff_biosecurity_score,
                item.visitor_restriction_level,
                item.avg_age,
                item.breed_type,
                item.health_score,
                item.previous_incidence,
                item.mortality_rate_prev,
                item.season,
                item.is_rainy_season
            ]).reshape(1, -1)
            
            # 使用标量器进行数据标准化
            features_scaled = scaler.transform(features)
            
            # 转换为PyTorch张量
            features_tensor = torch.FloatTensor(features_scaled)
            
            # 进行预测
            with torch.no_grad():
                prediction = model(features_tensor).item()
            
            # 确保预测值在合理范围内
            predicted_incidence = max(0.0, min(prediction, 100.0))
            
            # 获取风险评估和建议
            risk_level, confidence, recommendations = get_risk_assessment(predicted_incidence)
            
            predictions.append(PredictionResponse(
                predicted_incidence=round(predicted_incidence, 2),
                confidence=confidence,
                risk_level=risk_level,
                recommendations=recommendations
            ))
        
        return BatchPredictionResponse(predictions=predictions)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"批量预测过程中出错: {str(e)}")

# 模型信息端点
@app.get("/model/info")
async def model_info():
    return {
        "model_type": "神经网络",
        "input_features": 22,
        "hidden_layers": [64, 32, 16],
        "output": "猪群发病率预测",
        "framework": "PyTorch",
        "version": "1.0"
    }

# 启动时尝试加载模型
@app.on_event("startup")
async def startup_event():
    load_model_and_scaler()

# 主函数，用于本地运行
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )