import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# 간단한 신경망 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(5, 3)  # 5 입력 → 3 출력

    def forward(self, x):
        return self.fc(x)

# 모델 생성
model = SimpleModel()

# 가중치 확인 (Pruning 전)
print("Original Weights:\n", model.fc.weight)

# Pruning 적용 (L1 기반, 50% Pruning)
prune.l1_unstructured(model.fc, name="weight", amount=0.5)

# Mask 확인
print("Mask:\n", model.fc.weight_mask)

# Pruning 후 가중치 확인
print("Pruned Weights:\n", model.fc.weight)
