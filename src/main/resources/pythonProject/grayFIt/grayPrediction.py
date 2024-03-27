import numpy as np

import sys
import json


data_json_str = sys.argv[-1]  # 获取从 Java 传递过来的 JSON 字符串参数
# print(data_json_str)
# 解析 JSON 字符串为 Python 对象
data_array = np.array(json.loads(data_json_str))
data_lens = data_array.shape[0]
predictionData_lens = data_lens // 12

# 数据检验
## 计算级比
lambds = []
for i in range(1, data_lens):
    lambds.append(data_array[i - 1] / data_array[i])
## 计算区间
X_min = np.e ** (-2 / (data_lens + 1))
X_max = np.e ** (2 / (data_lens + 1))
## 检验
test = True
for lambd in lambds:
    if (lambd < X_min or lambd > X_max):
        test = False
# if test == False:
#         print('该数据不可以用灰色GM(1,1)模型')
# else:
#         print('该数据可以用灰色GM(1,1)模型')


# 构建灰色模型GM(1,1)
## 生成累加数列
data_add = []
for i in range(1, data_lens):
    data_add = data_array.cumsum()
## 生成紧邻均值序列
ds = []
zs = []
for i in range(1, data_lens):
    ds.append(data_array[i])
    zs.append(-1 / 2 * (data_add[i - 1] + data_add[i]))
## 生成矩阵和最小二乘法
B = np.array(zs).reshape(data_lens - 1, 1)
one = np.ones(data_lens - 1)
B = np.c_[B, one]  # 加上一列1
Y = np.array(ds).reshape(data_lens - 1, 1)
a, b = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
# print('a=' + str(a))
# print('b=' + str(b))


# 预测
def forecast(k):
    c = b / a
    return ((data_array[0] - c) * (np.e ** (-a * k)) + c).item()
data_1_for = []  # 累加预测值
data_0_for = []  # 原始预测值
data_1_for.append(forecast(0))
data_0_for.append(data_1_for[0])
for i in range(1, data_lens + predictionData_lens):  # 多预测 predictionData_lens 次
    data_1_for.append(float("{:.4f}".format(forecast(i))))
    data_0_for.append(float("{:.4f}".format(data_1_for[i] - data_1_for[i - 1])))
# print("原始数据为：\n", data_array)
# data_ = data_0_for[:-predictionData_lens]
# print('原始数据的预测值为：\n', data_)
predictionData = data_0_for[-predictionData_lens:]
# print('未来数据的预测值为：')
print(json.dumps(predictionData))


# # 模型检验
# ## 预测结果方差
# data_h = np.array(data_0_for[0:data_lens]).T
# sum_h = data_h.sum()
# mean_h = sum_h / data_lens
# S1 = np.sum((data_h - mean_h) ** 2) / data_lens
# ## 残差方差
# e = data_array - data_h
# e_sum = e.sum()
# e_mean = e_sum / data_lens
# S2 = np.sum((e - e_mean) ** 2) / data_lens
# ## 后验差比
# C = S2 / S1
# ## 结果
# if (C <= 0.35):
#     print('1级，效果好')
# elif (C <= 0.5 and C >= 0.35):
#     print('2级，效果合格')
# elif (C <= 0.65 and C >= 0.5):
#     print('3级，效果勉强')
# else:
#     print('4级，效果不合格')

# # 可视化
# plt.figure(figsize=(30, 30), dpi=100)
# x1 = np.linspace(1, 5, 5)
# x2 = np.linspace(1, 10, 10)
# plt.subplot(121)
# plt.title('x^0')
# plt.plot(x2, data_0_for, 'r--', marker='*')
# plt.scatter(x1, data_array, marker='^')
# plt.subplot(122)
# plt.title('x^1')
# plt.plot(x2, data_0_for, 'r--', marker='*')
# plt.scatter(x1, data_add, marker='^')
# plt.show()
