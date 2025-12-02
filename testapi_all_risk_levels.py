import requests
import json

# 猪发病率预测系统 - 各风险等级测试示例
# 风险等级划分标准：
# - 低风险：预测发病率 < 1.0%
# - 中等风险：预测发病率 1.0% - 3.0%
# - 高风险：预测发病率 3.0% - 5.0%
# - 极高风险：预测发病率 >= 5.0%

def test_predict(data, risk_level_name):
    """执行预测请求并打印结果"""
    url = "http://localhost:8000/predict"
    headers = {"Content-Type": "application/json"}
    
    print(f"\n{'='*50}")
    print(f"测试 {risk_level_name} 场景")
    print(f"{'='*50}")
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = response.json()
    print(f"预测结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

# 1. 低风险场景测试数据 (< 1.0%)
low_risk_data = {
    # 环境因素 - 基于训练数据低风险样本特征
    "temperature": 34.0,  # 温度较高：34°C，参考训练数据中的低风险样本特征
    "humidity": 42.0,     # 低湿度：42%，训练数据中低风险样本的典型湿度
    "ventilation_rate": 1800,  # 良好通风：1800，确保空气质量
    "ammonia_level": 2.3,      # 低氨水平：2.3，远低于有害水平
    "co2_level": 800,          # 低CO2：800，空气清新
    
    # 饲养管理因素 - 优秀条件（使用5分制，与训练数据一致）
    "pig_density": 0.2,      # 低密度：0.2，参考训练数据中最低密度
    "avg_feed_intake": 4.0,  # 高采食量：4.0，健康猪群表现
    "water_quality": 5,      # 优质水：5分（满分），与训练数据一致
    "feed_quality": 5,       # 优质饲料：5分（满分），与训练数据一致
    "bedding_condition": 5,  # 良好垫料：5分（满分），与训练数据一致
    
    # 生物安全因素 - 严格执行（使用5/9分制，与训练数据一致）
    "vaccination_rate": 100.0,    # 高接种率：100%
    "disinfection_frequency": 7,  # 高频消毒：每日7次
    "quarantine_days": 10,        # 严格检疫：10天，与训练数据一致
    "staff_biosecurity_score": 9, # 优秀生物安全：9分（训练数据最大值）
    "visitor_restriction_level": 5,  # 严格访客限制：5级（训练数据最大值）
    
    # 猪群特征 - 健康猪群
    "avg_age": 40,          # 年龄较小：40日龄，参考训练数据中低风险样本
    "breed_type": 1,         # 抗病品种：1号品种
    "health_score": 9,       # 健康评分：9分（训练数据最大值）
    
    # 历史数据 - 低风险历史
    "previous_incidence": 0.3,   # 低发病率历史：0.3%
    "mortality_rate_prev": 0.2,  # 低死亡率历史：0.2%
    
    # 季节性因素 - 适宜季节
    "season": 2,             # 夏季：2，与训练数据中低风险样本一致
    "is_rainy_season": 0     # 非雨季
}

# 2. 中等风险场景测试数据 (1.0% - 3.0%)
medium_risk_data = {
    # 环境因素 - 轻微问题
    "temperature": 26.0,     # 温度略高：26°C，开始有热应激风险
    "humidity": 70.0,        # 湿度偏高：70%，增加病原体传播风险
    "ventilation_rate": 1800,  # 通风一般：1800，基本满足需求
    "ammonia_level": 3.0,      # 氨水平中等：3.0，开始有刺激风险
    "co2_level": 1200,         # CO2偏高：1200，空气质量一般
    
    # 饲养管理因素 - 一般条件
    "pig_density": 0.8,      # 密度适中：0.8，略有拥挤
    "avg_feed_intake": 2.5,  # 采食量中等：2.5，略有下降
    "water_quality": 3,      # 水质一般：3分（与训练数据保持一致，使用5分制）
    "feed_quality": 3,       # 饲料质量一般：3分（与训练数据保持一致，使用5分制）
    "bedding_condition": 3,  # 垫料条件一般：3分（与训练数据保持一致，使用5分制）
    
    # 生物安全因素 - 基本执行
    "vaccination_rate": 90.0,     # 接种率中等：90%
    "disinfection_frequency": 3,  # 消毒频率一般：每日3次
    "quarantine_days": 14,        # 检疫天数：14天
    "staff_biosecurity_score": 7, # 生物安全评分：7分
    "visitor_restriction_level": 3,  # 访客限制：3级
    
    # 猪群特征 - 一般健康状况
    "avg_age": 150,          # 年龄较大：150日龄
    "breed_type": 2,         # 普通品种：2号品种
    "health_score": 7,       # 健康评分：7分，一般健康
    
    # 历史数据 - 中等风险历史
    "previous_incidence": 1.5,   # 发病率历史：1.5%
    "mortality_rate_prev": 1.2,  # 死亡率历史：1.2%
    
    # 季节性因素 - 一般季节
    "season": 2,             # 夏季：2
    "is_rainy_season": 0     # 非雨季
}

# 3. 高风险场景测试数据 (3.0% - 5.0%)
high_risk_data = {
    # 环境因素 - 较明显问题
    "temperature": 24.0,     # 温度略高：24°C，略微提高风险
    "humidity": 75.0,        # 湿度较高：75%，略微提高风险
    "ventilation_rate": 900,   # 通风较差：900，略微提高风险
    "ammonia_level": 8.5,      # 氨水平较高：8.5，略微提高风险
    "co2_level": 1600,         # CO2较高：1600，略微提高风险
    
    # 饲养管理因素 - 一般条件
    "pig_density": 1.2,      # 密度较高：1.2，略微提高风险
    "avg_feed_intake": 2.2,  # 采食量略低：2.2，略微提高风险
    "water_quality": 3,      # 水质一般：3分，与CSV保持一致
    "feed_quality": 3,       # 饲料质量略低：3分，略微提高风险
    "bedding_condition": 3,  # 垫料条件一般：3分，与CSV保持一致
    
    # 生物安全因素 - 基本执行但有不足
    "vaccination_rate": 84.0,     # 接种率略低：84%，略微提高风险
    "disinfection_frequency": 2,  # 消毒频率一般：每日2次
    "quarantine_days": 4,         # 检疫天数短：4天，略微提高风险
    "staff_biosecurity_score": 6, # 生物安全评分：6分，略微提高风险
    "visitor_restriction_level": 2,  # 访客限制松：2级，略微提高风险
    
    # 猪群特征 - 健康状况一般
    "avg_age": 170,          # 年龄较大：170日龄，略微提高风险
    "breed_type": 3,         # 易感品种：3号品种
    "health_score": 6,       # 健康评分：6分，略微提高风险
    
    # 历史数据 - 高风险历史
    "previous_incidence": 4.0,   # 高发病率历史：4.0%
    "mortality_rate_prev": 2.2,  # 死亡率历史：2.2%，略微提高风险
    
    # 季节性因素 - 一般季节
    "season": 2,             # 春季：2，略微提高风险
    "is_rainy_season": 0     # 非雨季
}

# 4. 极高风险场景测试数据 (>= 5.0%)
extreme_risk_data = {
    # 环境因素 - 严重问题
    "temperature": 32.0,     # 高温：32°C，严重热应激
    "humidity": 85.0,        # 极高湿度：85%，非常潮湿
    "ventilation_rate": 800,   # 严重通风不足：800
    "ammonia_level": 8.0,      # 严重高氨：8.0，强烈刺激
    "co2_level": 3000,         # CO2严重超标：3000
    
    # 饲养管理因素 - 恶劣条件
    "pig_density": 1.8,      # 极度拥挤：1.8
    "avg_feed_intake": 1.5,  # 采食量极低：1.5，食欲严重下降
    "water_quality": 1,      # 水质极差：1分，严重污染
    "feed_quality": 2,       # 饲料质量极差：2分，严重营养不足
    "bedding_condition": 1,  # 垫料极差：1分，极度潮湿肮脏
    
    # 生物安全因素 - 严重不足
    "vaccination_rate": 65.0,     # 接种率极低：65%
    "disinfection_frequency": 1,  # 很少消毒：每日1次
    "quarantine_days": 5,         # 几乎不检疫：5天
    "staff_biosecurity_score": 4, # 生物安全差：4分
    "visitor_restriction_level": 1,  # 几乎无限制：1级
    
    # 猪群特征 - 健康状况差
    "avg_age": 200,          # 年龄很大：200日龄
    "breed_type": 4,         # 高易感品种：4号品种
    "health_score": 5,       # 健康评分低：5分，健康状况差
    
    # 历史数据 - 极高风险历史
    "previous_incidence": 7.0,   # 极高发病率历史：7.0%
    "mortality_rate_prev": 5.5,  # 极高死亡率历史：5.5%
    
    # 季节性因素 - 极高风险季节
    "season": 1,             # 冬季：1
    "is_rainy_season": 1     # 雨季
}

# 执行所有测试
if __name__ == "__main__":
    print("猪发病率预测系统 - 各风险等级测试")
    print("="*50)
    
    # 测试低风险场景
    test_predict(low_risk_data, "低风险")
    
    # 测试中等风险场景
    test_predict(medium_risk_data, "中等风险")
    
    # 测试高风险场景
    test_predict(high_risk_data, "高风险")
    
    # 测试极高风险场景
    test_predict(extreme_risk_data, "极高风险")
    
    print("\n所有风险等级测试完成！")
