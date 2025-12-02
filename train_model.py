import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据加载和预处理
def load_and_preprocess_data(file_path):
    # 加载CSV数据
    df = pd.read_csv(file_path)
    
    # 分离特征和目标变量
    X = df.iloc[:, :-1].values  # 前22列是特征
    y = df.iloc[:, -1].values   # 最后一列是目标变量(发病率)
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)  # 转换为列向量
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    return {
        'X_train': X_train_tensor,
        'X_test': X_test_tensor,
        'y_train': y_train_tensor,
        'y_test': y_test_tensor,
        'scaler': scaler
    }

# 2. 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=22, hidden_sizes=[64, 32, 16], output_size=1):
        super(NeuralNetwork, self).__init__()
        
        # 创建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # 添加Dropout防止过拟合
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        # 组合所有层
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 3. 训练模型
def train_model(model, train_data, learning_rate=0.001, epochs=500, batch_size=32):
    # 准备数据
    X_train = train_data['X_train']
    y_train = train_data['y_train']
    X_test = train_data['X_test']
    y_test = train_data['y_test']
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 训练历史记录
    train_losses = []
    test_losses = []
    
    # 开始训练
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
        
        # 计算平均训练损失
        train_loss = running_loss / len(X_train)
        train_losses.append(train_loss)
        
        # 在测试集上评估
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_losses.append(test_loss)
        
        # 更新学习率
        scheduler.step(test_loss)
        
        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    # 计算最终的评估指标
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = criterion(y_pred, y_test).item()
        rmse = np.sqrt(mse)
        
        # 计算R²
        y_test_np = y_test.numpy()
        y_pred_np = y_pred.numpy()
        ss_res = np.sum((y_test_np - y_pred_np) **2)
        ss_tot = np.sum((y_test_np - np.mean(y_test_np)) **2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f'\n模型评估结果:')
    print(f'均方根误差 (RMSE): {rmse:.6f}')
    print(f'决定系数 (R²): {r2:.6f}')
    
    # 绘制训练和测试损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, label='训练损失')
    plt.plot(range(1, epochs+1), test_losses, label='测试损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('模型训练过程')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
    return {
        'rmse': rmse,
        'r2': r2,
        'train_losses': train_losses,
        'test_losses': test_losses
    }

# 4. 保存模型和预处理对象
def save_model_and_scaler(model, scaler, model_dir='models'):
    # 创建目录（如果不存在）
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(model_dir, 'pig_disease_model.pth'))
    
    # 保存标量器
    import joblib
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    print(f'模型和标量器已保存到 {model_dir} 目录')

# 主函数
def main():
    print('开始加载和预处理数据...')
    train_data = load_and_preprocess_data('pig_disease_training_data.csv')
    
    print('初始化神经网络模型...')
    model = NeuralNetwork(input_size=22, hidden_sizes=[64, 32, 16], output_size=1)
    
    print('开始训练模型...')
    results = train_model(
        model,
        train_data,
        learning_rate=0.001,
        epochs=500,
        batch_size=32
    )
    
    print('保存模型和预处理对象...')
    save_model_and_scaler(model, train_data['scaler'])
    
    print('模型训练和保存完成!')

if __name__ == '__main__':
    main()