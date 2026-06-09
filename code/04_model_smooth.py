import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
df = pd.read_csv("../data/processed_rmb_rate.csv", index_col="日期", parse_dates=["日期"])
target_currency = "美元"
series = df[target_currency]

# 划分数据集
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# 霍尔特线性指数平滑（适配带长期趋势的汇率时序数据）
model = Holt(train)
model_fit = model.fit()
print(f"\n===== 指数平滑模型训练结果 =====")
print(model_fit.summary())

# 预测
predictions = model_fit.forecast(steps=len(test))

# 模型评估
mae = mean_absolute_error(test, predictions)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f"\n===== 指数平滑 模型评估指标 =====")
print(f"平均绝对误差 MAE: {mae:.4f}")
print(f"均方误差 MSE: {mse:.4f}")
print(f"根均方误差 RMSE: {rmse:.4f}")

# 绘图
plt.figure(figsize=(14, 6))
plt.plot(test.index, test.values, label="真实汇率值", color="#1f77b4", linewidth=1.2)
plt.plot(test.index, predictions.values, label="指数平滑预测值", color="#2ca02c", linewidth=1.2, linestyle="--")
plt.title(f"指数平滑模型 {target_currency}兑人民币汇率 预测结果对比")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"../result/{target_currency}_smooth_predict.png", dpi=300, bbox_inches="tight")
plt.show()
print(f"\n✅ 指数平滑预测图表已保存至 result 文件夹")