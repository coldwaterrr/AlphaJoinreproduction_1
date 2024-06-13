import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ValueNet, self).__init__()
        self.dim = in_dim
        self.layer1 = nn.Sequential(nn.Linear(in_dim, 2048), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(128, out_dim), nn.Softmax(dim = 0))
        # self.layer5 = nn.Sequential(nn.Linear(32, out_dim), nn.ReLU(True))

    def forward(self, x):
        # x = x.reshape(-1, self.dim)
        x = self.layer1(x)
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.layer2(x)
        # x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.layer3(x)
        # x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.layer4(x)
        # x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.layer5(x)
        return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class ValueNet(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ValueNet, self).__init__()
#         self.dim = in_dim
#         self.layer1 = nn.Sequential(
#             nn.Linear(in_dim, 2048),
#             # nn.BatchNorm1d(2048),
#             nn.ReLU(True),
#             nn.Dropout(0.5)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Linear(2048, 1024),
#             # nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.5)
#         )
#         self.layer3 = nn.Sequential(
#             nn.Linear(1024, 512),
#             # nn.BatchNorm1d(512),
#             nn.ReLU(True),
#             nn.Dropout(0.4)
#         )
#         self.layer4 = nn.Sequential(
#             nn.Linear(512, 128),
#             # nn.BatchNorm1d(128),
#             nn.ReLU(True),
#             nn.Dropout(0.3)
#         )
#         self.layer5 = nn.Linear(128, out_dim)
#
#     def forward(self, x):
#         # if x.dim() == 1:
#         #     x = x.unsqueeze(0)  # 将1D输入转换为2D输入
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x



# Training loop (simplified)
# for epoch in range(num_epochs):
#     output = model(input_data)
#     loss = loss_func(output, target_data)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
