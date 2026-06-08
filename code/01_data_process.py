import pandas as pd
import numpy as np

# 读取官方人民币汇率Excel文件
file_path = r"C:\Users\21182\Desktop\exchange-rate-forecast\data\rmb_exchange_rate.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet0")

# 转换日期列为时间索引（时序项目核心步骤）
df["日期"] = pd.to_datetime(df["日期"])
df = df.set_index("日期")

# 按时间升序排序（原始数据是倒序，时序分析必须升序）
df = df.sort_index(ascending=True)

# 查看基础信息
print("===== 数据集基本信息 =====")
print(f"数据时间范围：{df.index.min()} 至 {df.index.max()}")
print(f"数据总行数：{len(df)}")
print(f"包含币种：{df.columns.tolist()}")
print("\n数据统计描述：")
print(df.describe())

# 缺失值处理：时序数据推荐前向填充，剔除剩余空值
df = df.ffill()
df = df.dropna()

# 保存处理后的数据，方便后续调用
df.to_csv("../data/processed_rmb_rate.csv")
print("\n✅ 数据预处理完成，已保存为 processed_rmb_rate.csv")