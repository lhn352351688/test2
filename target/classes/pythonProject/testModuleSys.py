import sys
import json

gpus_json_str = sys.argv[-1]  # 获取从 Java 传递过来的 JSON 字符串参数

# 解析 JSON 字符串为 Python 对象
gpus_list = json.loads(gpus_json_str)

print(type(gpus_list))  # 打印类型
print(gpus_list)         # 打印原始列表数据

print("Hello")