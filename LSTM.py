import torch
import torch.nn as nn
import torch.optim as optim

# device 설정 (CUDA가 가능하면 GPU 사용, 아니면 CPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM 층
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 완전 연결층
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 초기 은닉 상태와 셀 상태를 설정
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM에 입력 데이터를 전달하고 출력값과 최종 상태를 받음
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 시점의 은닉 상태를 사용하여 예측값을 계산
        out = self.fc(out[:, -1, :])
        return out

# 모델 파라미터 설정
input_dim = 1  # 특징 수
hidden_dim = 64
output_dim = 2  # 레이블 수
num_layers = 2

# 모델 생성
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
model = model.to(device)

# 손실 함수와 최적화 함수 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)