import torch
from torch.utils.data import DataLoader
from src.models.cbma import CBMA
from src.utils.dataset import DDOSDataset
from src.utils.metrics import calculate_metrics

def train(config):
    # 1. 数据加载
    train_set = DDOSDataset("data/processed/train.csv")
    train_loader = DataLoader(train_set, batch_size=config['batch_size'])
    
    # 2. 初始化模型
    model = CBMA(input_dim=config['input_dim'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    
    # 3. 训练循环
    for epoch in range(config['epochs']):
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
        # 验证集评估
        val_metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch}: Val Accuracy={val_metrics['accuracy']:.2f}")

def evaluate(model, dataloader):
    """实现论文3.2节的评估指标"""
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())
    
    return calculate_metrics(y_true, y_pred)  # 实现论文公式2.11-2.14
