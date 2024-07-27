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
from scipy.spatial import distance
import time
import tracemalloc
from torchinfo import summary

start = time.perf_counter()

# 启动内存跟踪
tracemalloc.start()

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
            key_size = num_inputs if i == 0 else num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, key_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        out = x
        for i in range(len(self.network)):
            out = self.network[i](out)

        return out


class TCANModel(BaseEstimator, RegressorMixin, nn.Module):
    def __init__(self, input_size, output_size, num_channels, seq_length, n_epoch=100, batch_size=64, lr=0.001,
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
        # X = self.scaler_X.fit_transform(X)
        #
        # y = y.reshape(-1, 1)
        #
        # X_3d = []
        # y_3d = []
        # for i in range(X.shape[0] - self.seq_length + 1):
        #     X_3d.append(X[i: i + self.seq_length, :])
        #     y_3d.append(y[i + self.seq_length - 1: i + self.seq_length, :])
        #
        # X_3d = np.stack(X_3d, 1)
        # X_3d = torch.tensor(X_3d)
        # X_3d = X_3d.permute(1, 2, 0)
        # X_3d = X_3d.numpy()
        # y_3d = np.stack(y_3d, 1)
        # y_3d = torch.tensor(y_3d)
        # y_3d = y_3d.permute(1, 2, 0)
        # y_3d = y_3d.numpy()
        dataset = MyDataset(torch.tensor(X, dtype=torch.float32, device=device),
                            torch.tensor(y, dtype=torch.float32, device=device), '3D')
        self.model.train()
        epoch_start_time = time.perf_counter()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.permute(0, 1, 2)
                batch_y = batch_y.permute(0, 1, 2)

                batch_y = batch_y.squeeze(1)

                self.optimizer.zero_grad()

                output = self.model(batch_X)

                #output1 = output[:, :, -1]
                output = self.linear(output[:, :, -1])

                loss = self.criterion(output, batch_y)

                self.loss_hist[-1] += loss.item()

                loss.backward()

                self.optimizer.step()
            epoch_end_time = time.perf_counter()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_start_time = epoch_end_time
            print('Epoch:{}, Loss:{}, Time taken: {:.2f} sec'.format(i + 1, self.loss_hist[-1], epoch_duration))
        print('Optimization finished')

            # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
        # plt.cla()
        # x1 = range(1, self.n_epoch+1)
        # print(x1)
        # y1 = self.loss_hist
        # print(y1)
        # plt.title('Train loss ', fontsize=20)
        # plt.plot(x1, y1, '.-')
        # plt.xlabel('epoches', fontsize=20)
        # plt.ylabel('Train loss', fontsize=20)
        # plt.grid()
        # plt.show()

        return self

    def predict(self, X):

        # X = self.scaler_X.transform(X)
        #
        # # 转化为三维再预测
        # X_3d = []
        #
        # for i in range(X.shape[0] - seq_length + 1):
        #     X_3d.append(X[i: i + seq_length, :])
        # X_3d = np.stack(X_3d, 1)
        # X_3d = torch.tensor(X_3d)
        # X_3d = X_3d.permute(1, 2, 0)
        # X_3d = X_3d.numpy()
        # X = torch.tensor(X_3d, dtype=torch.float32, device=device).permute(0, 1, 2)

        self.model.eval()
        with torch.no_grad():
            y = self.model(X)
            y = self.linear(y[:, :, -1])

            # 放上cpu转为numpy
            y = y.cpu().numpy()

        return y

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

SEQ_LEN = 40

data = pd.read_csv('Debutanizer_Data.txt', sep='\s+')

data = data.values

# TRAIN_SIZE = 2390
TRAIN_SIZE = 1000

x_temp = data[:, 0:7]      # 取data中第1维（即列）的第0到第6个元素，第0维取全部
y_temp = data[:, 7]        #  第0维取全部，第1维取第7个元素

# x_new = np.zeros([data.shape[0], 8])
# x_new[:,:7] = x_temp
# x_new[0,7] = y_temp[0]
# x_new[1:,7] = y_temp[:2392]

history_x = x_temp[:TRAIN_SIZE, :]
history_y = y_temp[:TRAIN_SIZE]
history_y = history_y.reshape(-1, 1)

test_X = x_temp[TRAIN_SIZE:, :]
y_test = y_temp[TRAIN_SIZE:]
y_test = y_test.reshape(-1, 1)
window = 30
seq_len = 40
y_pre = []
col = test_X.shape[1]
# tcan_model = TCANModel(input_size=7, output_size=1, num_channels=[40, 40, 40], seq_length=SEQ_LEN, n_epoch=240, batch_size=64,
#                 lr=0.001, seed=1024)
#
# # 显示模型摘要
# tcan_model.show_model_summary()                                                   # epoch论文中是100
model = TCANModel(input_size=7, output_size=1, num_channels=[40, 40, 40], seq_length=SEQ_LEN, n_epoch=100, batch_size=64, lr=0.001,seed=1024, device=device)  # 初始是100，改为240
for n in range(0, len(test_X[:, 0])):
    # 测试集先选出来一个Xq,和每个历史数据算欧式距离
    dist = []
    for i in range(0, len(history_x[:, 0])):
        #1欧式距离
        dist.append(distance.euclidean(test_X[n], history_x[i]))
        #2余弦相似度
        #x1 = torch.FloatTensor(test_X[n]).reshape(-1, 1)
        #x2 = torch.FloatTensor(history_x[i]).reshape(-1, 1)
        #similarity = torch.cosine_similarity(x1, x2, dim=0, eps=1e-08)
        #similarity = cosine_similarity(test_X[n], history_x[i])
        #dist.append(similarity)
    D0 = np.array(dist, dtype=np.float64)
    D1 = D0.reshape(len(history_x[:, 0]), 1)
    x_DB1 = np.append(history_x, D1, axis=1)
    # x_DB1 = np.append(x_DB1, history_y, axis=1)
    n_DB = x_DB1[np.argsort(x_DB1[:, col])]
    data_r = n_DB[0:window, :]  # 筛选前30个相关样本作为回归数据集X1,X2,X3,....,X30
    # 把欧式距离那一列删除才能在历史数据中查找
    data_ = np.delete(data_r, col, axis=1)
    # 结合时间 将Xi的前29个时刻找出
    input = []
    output = []
    for m in range(0, window):
        #hang = data_[m, :]
        index = history_x.tolist().index(data_[m, :].tolist())
        if index < 39:
            input.append(history_x[0:40, :])
            output.append(history_y[39, :].reshape(-1, 1))
        # li = history_x[index - seq_len + 1:index + 1, :]
        if index >= 39:
            input.append(history_x[index - seq_len + 1:index + 1, :])
            output.append(history_y[index, :].reshape(-1, 1))
    input = np.array(input, dtype=np.float64)
    input = np.transpose(input, (0, 2, 1))
    # output = np.array(output, dtype=np.float64)
    output = np.stack(output, 1)
    output = np.transpose(output, (1, 2, 0))
    # x_r = data_r[:, :col]
    # y_r = data_r[:, -1].reshape(30, 1)
    history_x = np.append(history_x, test_X[n].reshape(1, col), axis=0)
    history_y = np.append(history_y, y_test[n].reshape(1, 1), axis=0)
    # Xq的前24个时刻数据和Xq作为测试
    id = history_x.tolist().index(test_X[n].tolist())
    test_x = []
    test_x.append(history_x[id-seq_len+1:id+1, :])
    test_x = np.stack(test_x, 1)
    test_x = np.transpose(test_x, (1, 2, 0))
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device).permute(0, 1, 2)
    y_pre.append(model.fit(input, output).predict(test_x))

y_pred = np.array(y_pre)
y_pred = y_pred.reshape(len(y_pred), 1)

end = time.perf_counter()
# 获取内存分配信息
current, peak = tracemalloc.get_traced_memory()
print(f"当前内存分配: {current / 10 ** 6} MB")
print(f"峰值内存分配: {peak / 10 ** 6} MB")
# 停止内存跟踪
tracemalloc.stop()
# mdl = TCANModel(input_size=7, output_size=1, num_channels=[30, 30, 30], seq_length=SEQ_LEN, n_epoch=240, batch_size=64,
#                 lr=0.001, seed=1024, device=device).fit(train_X, y_train)
#
# y_train_pred = mdl.predict(train_X, seq_length=SEQ_LEN)
# plt.figure()
# plt.plot(range(len(y_train_pred)), y_train_pred, color='b', label='y_trainpre')
# plt.plot(range(len(y_train_pred)), train_y, color='r', label='y_true')
# plt.legend()
# plt.show()
# rmse = math.sqrt(mean_squared_error(train_y, y_train_pred))
# print('\nMSE：', mean_squared_error(train_y, y_train_pred))
# print('\nRMSE:', rmse)
# print('\n相关系数：', r2_score(train_y, y_train_pred))
#
# y_pred = mdl.predict(test_X, seq_length=SEQ_LEN)
# np.savetxt('JIT-RTA-TCN y_pred.txt', y_pred)
# np.savetxt('y_true.txt', y_test)

plt.figure(figsize=(10, 5), dpi=130)
y1 = y_pred
y2 = y_test
plt.plot(y_pred, color='b', label='y_pred', linewidth=1.5)
plt.plot(y_test, color='r', label='y_real', linewidth=1.5)
# plt.text(600, 0.8, '(d)', fontsize=12)
plt.xlim(0, len(test_X[:, 0]))
plt.ylim(0, 1)
plt.xlabel('Sample number', fontsize=12, fontweight='bold')
plt.ylabel('Output value', fontsize=12, fontweight='bold')
plt.title('JITL-RTA-TCN', fontsize=16)
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')
plt.legend()
#plt.savefig('G:\experiment\TCN\TA-TCN.svg')
plt.show()
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print('\nMSE：', mean_squared_error(y_test, y_pred))
print('\nRMSE:', rmse)
print('\n相关系数：', r2_score(y_test, y_pred))
print("运行耗时", end-start)
error = torch.abs(torch.tensor(y_test - y_pred))
# np.savetxt('JIT-RTA-TCN error.txt', error)