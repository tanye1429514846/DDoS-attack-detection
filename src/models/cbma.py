import torch
import torch.nn as nn

class CBMA(nn.Module):
    """CNN-BiLSTM with Attention (论文第2.3节)"""
    def __init__(self, input_dim=77, num_classes=2):
        super().__init__()
        # 1D-CNN 分支 (论文图8)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=2),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        # BiLSTM 分支 (论文2.3.2节)
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            bidirectional=True
        )
        
        # 自注意力机制 (论文2.3.3节公式2.8-2.10)
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # 分类器
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # 输入x形状: (batch_size, seq_len, input_dim)
        
        # CNN分支
        cnn_out = self.cnn(x.permute(0,2,1))  # 转换为(batch, channels, seq)
        cnn_out = cnn_out.permute(0,2,1)       # 恢复时序维度
        
        # BiLSTM分支
        lstm_out, _ = self.bilstm(x)
        
        # 特征融合
        fused = torch.cat([cnn_out, lstm_out], dim=-1)
        
        # 注意力权重
        attn_weights = self.attention(fused)
        weighted = torch.sum(attm_weights * fused, dim=1)
        
        return self.classifier(weighted)
