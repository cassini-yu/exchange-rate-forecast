import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据并构造时序特征
df = pd.read_csv("../data/processed_rmb_rate.csv", index_col="日期", parse_dates=["日期"])
target_currency = "美元"
data = df[target_currency].copy().to_frame()

# 构造时序滞后特征（机器学习时序建模核心特征工程）
data["lag_1"] = data[target_currency].shift(1)  # 前1天汇率
data["lag_7"] = data[target_currency].shift(7)  # 前7天汇率
data = data.dropna()

# 特征与标签划分
X = data[["lag_1", "lag_7"]]
y = data[target_currency]

# 划分训练集、测试集（和前两个模型保持一致，保证对比公平）
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 训练随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 模型评估
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\n===== 随机森林 模型评估指标 =====")
print(f"平均绝对误差 MAE: {mae:.4f}")
print(f"均方误差 MSE: {mse:.4f}")
print(f"根均方误差 RMSE: {rmse:.4f}")

# 多模型结果汇总保存
result_df = pd.DataFrame({
    "日期": y_test.index,
    "真实汇率值": y_test.values,
    "随机森林预测值": y_pred
})
result_df.to_csv(f"../result/{target_currency}_model_eval_result.csv", index=False)

# 绘制预测结果图
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test.values, label="真实汇率值", color="#1f77b4", linewidth=1.2)
plt.plot(y_test.index, y_pred, label="随机森林预测值", color="#9467bd", linewidth=1.2, linestyle="--")
plt.title(f"随机森林模型 {target_currency}兑人民币汇率 预测结果对比")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"../result/{target_currency}_rf_predict.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\n✅ 所有模型实验完成，指标表格与图表已全部保存至 result 文件夹")