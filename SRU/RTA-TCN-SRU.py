import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils import weight_norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tracemalloc
import time
from torchinfo import summary

start = time.perf_counter()

device = torch.device('cpu')


class MyDataset(Dataset):

    # Initialization
    def __init__(self, data, label, mode='2D'):
        self.data, self.label, self.mode = data, label, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :]
        elif self.mode == '3D':
            return self.data[index, :, :], self.label[index, :, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[0]


# 这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。这个函数就是第一个数据到倒数第chomp_size的数据，
# 这个chomp_size就是padding的值。比方说输入数据是5，padding是1，那么会产生6个数据没错吧，那么就是保留前5个数字
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        # 表示对继承自父类属性进行初始化
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        tensor.contiguous()会返回有连续内存的相同张量
        有些tensor并不是占用一整块内存，而是由不同的数据块组成
        tensor的view()操作依赖于内存是整块的，这时只需要执行
        contiguous()函数，就是把tensor变成在内存中连续分布的形式
        本函数主要是增加padding方式对卷积后的张量做切边而实现因果卷积
        """
        return x[:, :, :-self.chomp_size].contiguous()


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_value = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)  # 开平方根

    def forward(self, input):
        #  input(batch_size,in_channels,seq_len)
        mask = np.array([[1 if i > j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        mask = torch.ByteTensor(mask)
        mask = mask.bool()
        # seq_len = input.size(2)
        # # 使用 input.device 确保 mask 和 input 在同一个设备上
        # mask = torch.tril(torch.ones(seq_len, seq_len, device=input.device, dtype=torch.bool))

        input = input.permute(0, 2, 1)
        keys = self.linear_keys(input)
        query = self.linear_query(input)
        value = self.linear_value(input)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))  ## keys第二维和第三维维度第调换，bmm:两个矩阵相乘
        temp.data.masked_fill_(mask, -float('inf'))  ### 只要下三角，其余用0填充

        weight_temp = F.softmax(temp / self.sqrt_key_size, dim=1)
        value_attentioned = torch.bmm(weight_temp, value).permute(0, 2, 1)
        return value_attentioned
        #   value_attentioned 就是 Sa ,weight_temp 就是 softmax之后的Wa


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, key_size, stride, padding, dilation, dropout):
        super(TemporalBlock, self).__init__()

        self.attention = AttentionBlock(n_inputs, key_size, n_inputs)

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        out_attn = self.attention(x)
        out_attn = out_attn + x
        out = self.net(out_attn)

        #weight_x = F.softmax(attn_weight.sum(dim=2), dim=1)
        #en_res_x = weight_x.unsqueeze(2).repeat(1, 1, x.size(1)).transpose(1, 2) * x
        #en_res_x = en_res_x if self.downsample is None else self.downsample(en_res_x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TemporalConvAttnNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=4, dropout=0.2):
        super(TemporalConvAttnNet, self).__init__()
        layers = []
        num_levels = 3
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数,输入层通道为1，隐含层是25。
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            key_size = 6 if i == 0 else num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, key_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        out = x
        for i in range(len(self.network)):
            out = self.network[i](out)

        return out


class TCANModel(BaseEstimator, RegressorMixin):
    def __init__(self, input_size, output_size, num_channels, seq_length, n_epoch=240, batch_size=64, lr=0.001,
                 device=device, seed=1024):
        super(TCANModel, self).__init__()
        torch.manual_seed(seed)

        self.num_channels = num_channels
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seq_length = seq_length
        self.seed = seed
        self.input_size = input_size
        self.output_size = output_size
        self.n_epoch = n_epoch

        self.scaler_X = StandardScaler()
        self.loss_hist = []

        self.model = TemporalConvAttnNet(input_size, num_channels, dropout=0.2).to(device)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='mean')

    def show_model_summary(self):
        """ 显示模型的结构和参数摘要 """
        summary(self.model, input_size=(1, self.input_size, self.seq_length))

    def fit(self, X, y):
        X = self.scaler_X.fit_transform(X)

        y = y.reshape(-1, 1)

        X_3d = []
        y_3d = []
        for i in range(X.shape[0] - self.seq_length + 1):
            X_3d.append(X[i: i + self.seq_length, :])
            y_3d.append(y[i + self.seq_length - 1: i + self.seq_length, :])

        X_3d = np.stack(X_3d, 1)
        X_3d = torch.tensor(X_3d)
        X_3d = X_3d.permute(1, 2, 0)
        X_3d = X_3d.numpy()
        y_3d = np.stack(y_3d, 1)
        y_3d = torch.tensor(y_3d)
        y_3d = y_3d.permute(1, 2, 0)
        y_3d = y_3d.numpy()
        dataset = MyDataset(torch.tensor(X_3d, dtype=torch.float32, device=device),
                            torch.tensor(y_3d, dtype=torch.float32, device=device), '3D')
        self.model.train()

        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.permute(0, 1, 2)
                batch_y = batch_y.permute(0, 1, 2)

                batch_y = batch_y.squeeze(1)

                self.optimizer.zero_grad()

                output = self.model(batch_X)

                output = self.linear(output[:, :, -1])

                loss = self.criterion(output, batch_y)

                self.loss_hist[-1] += loss.item()

                loss.backward()

                self.optimizer.step()

            print('Epoch:{}, Loss:{}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished')

        plt.cla()
        x1 = range(1, self.n_epoch+1)
        print(x1)
        y1 = self.loss_hist
        print(y1)
        plt.title('Train loss ', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoches', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.grid()
        plt.show()

        return self

    def predict(self, X, seq_length):

        X = self.scaler_X.transform(X)

        # 转化为三维再预测
        X_3d = []

        for i in range(X.shape[0] - seq_length + 1):
            X_3d.append(X[i: i + seq_length, :])
        X_3d = np.stack(X_3d, 1)
        X_3d = torch.tensor(X_3d)
        X_3d = X_3d.permute(1, 2, 0)
        X_3d = X_3d.numpy()
        X = torch.tensor(X_3d, dtype=torch.float32, device=device).permute(0, 1, 2)

        self.model.eval()
        with torch.no_grad():
            y = self.model(X)
            y = self.linear(y[:, :, -1])

            # 放上cpu转为numpy
            y = y.cpu().numpy()

        return y


# SEQ_LEN = 40
SEQ_LEN = 25

data = pd.read_csv('SRU_data.txt', sep='\s+')
data = data.values
# data = data.values[:3000,:]
#data = data[:6000, :]

TRAIN_SIZE = 7039 + SEQ_LEN - 1
# TRAIN_SIZE = 4976 + SEQ_LEN - 1
# TRAIN_SIZE = 1000

x_temp = data[:, 0:5]      # 取data中第1维（即列）的第0到第6个元素，第0维取全部
y_temp = data[:, 5]        #  第0维取全部，第1维取第7个元素

# x_new = np.zeros([data.shape[0], 6])
# x_new[:, :5] = x_temp
# x_new[0, 5] = y_temp[0]
# x_new[1:, 5] = y_temp[:2999]

# train_X = x_new[:TRAIN_SIZE, :]
# y_train = y_temp[:TRAIN_SIZE]
# train_y = y_train[SEQ_LEN - 1:TRAIN_SIZE]
#
# test_X = x_new[TRAIN_SIZE-SEQ_LEN+1:, :]
# y_test = y_temp[TRAIN_SIZE:]
# y_test = y_test.reshape(-1, 1)

train_X = x_temp[:TRAIN_SIZE, :]
y_train = y_temp[:TRAIN_SIZE]
train_y = y_train[SEQ_LEN-1:TRAIN_SIZE]

test_X = x_temp[TRAIN_SIZE-SEQ_LEN+1:, :]
y_test = y_temp[TRAIN_SIZE:]
y_test = y_test.reshape(-1, 1)

#np.savetxt('y_true.txt', y_test)

# # 创建模型实例
# tcan_model = TCANModel(input_size=5, output_size=1, num_channels=[40, 40, 40], seq_length=SEQ_LEN, n_epoch=240, batch_size=64,
#                 lr=0.001, seed=1024)
#
# # 显示模型摘要
# tcan_model.show_model_summary()

mdl = TCANModel(input_size=5, output_size=1, num_channels=[40, 40, 40], seq_length=SEQ_LEN, n_epoch=240, batch_size=64,
                lr=0.001, seed=1024, device=device).fit(train_X, y_train)   # input_size=6 是加y               # epoch=240

# 启动内存跟踪
tracemalloc.start()

y_train_pred = mdl.predict(train_X, seq_length=SEQ_LEN)
end = time.perf_counter()

# 获取内存分配信息
current, peak = tracemalloc.get_traced_memory()
print(f"当前内存分配: {current / 10 ** 6} MB")
print(f"峰值内存分配: {peak / 10 ** 6} MB")
# 停止内存跟踪
tracemalloc.stop()

plt.figure()
plt.plot(range(len(y_train_pred)), y_train_pred, color='b', label='y_trainpre')
plt.plot(range(len(y_train_pred)), train_y, color='r', label='y_true')
plt.legend()
plt.show()
rmse = math.sqrt(mean_squared_error(train_y, y_train_pred))
print('\nMSE：', mean_squared_error(train_y, y_train_pred))
print('\nRMSE:', rmse)
print('\n相关系数：', r2_score(train_y, y_train_pred))

y_pred = mdl.predict(test_X, seq_length=SEQ_LEN)
# np.savetxt('TA-TCN_y_pred.txt', y_pred)
np.savetxt('1000 OFFLINE_y_pred.txt', y_pred)

plt.figure(figsize=(10, 5), dpi=130)
y1 = y_pred
y2 = y_test
plt.plot(y_pred, color='b', label='y_pred', linewidth=1.5)
plt.plot(y_test, color='r', label='y_real', linewidth=1.5)
plt.text(800, 0.4, '(d)', fontsize=12)
plt.xlim(0, 2000)
plt.ylim(0, 1)
plt.xlabel('Sample number', fontsize=12, fontweight='bold')
plt.ylabel('Output value', fontsize=12, fontweight='bold')
plt.title('TA-TCN', fontsize=16)
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')
plt.legend()
#plt.savefig('G:\experiment\Sulfur Recovery Unit_Data\TA-TCN.svg')
plt.show()
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print('\nMSE：', mean_squared_error(y_test, y_pred))
print('\nRMSE:', rmse)
print('\n相关系数：', r2_score(y_test, y_pred))
error = torch.abs(torch.tensor(y_test - y_pred))
# np.savetxt('TA-TCN error.txt', error)
np.savetxt('OFFLINE error.txt', error)
print("总耗时: ", end - start)
