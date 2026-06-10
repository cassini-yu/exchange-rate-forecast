import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 全局配置
warnings.filterwarnings("ignore")
plt.switch_backend("Agg")  # 后台绘图，不弹出窗口
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 路径与参数配置
DATA_PATH = "c:/Users/21182/Desktop/exchange-rate-forecast/data/processed_rmb_rate.csv"
RESULT_ROOT = "c:/Users/21182/Desktop/exchange-rate-forecast/result"
TRAIN_RATIO = 0.8
# 随机森林固定参数，保证复现性
RF_ESTIMATORS = 100
RANDOM_SEED = 42

def run_rf_single_currency(currency_name, ts_series):
    """单币种随机森林回归建模、预测、结果保存"""
    # 数据清洗
    data = ts_series.copy().to_frame()
    data.columns = [currency_name]

    # 构造滞后特征：lag1、lag7
    data["lag_1"] = data[currency_name].shift(1)
    data["lag_7"] = data[currency_name].shift(7)
    data = data.dropna()

    # 数据量校验
    if len(data) < 50:
        print(f"{currency_name} 有效数据过少，跳过")
        return

    # 特征、标签划分
    X = data[["lag_1", "lag_7"]]
    y = data[currency_name]

    # 划分训练集、测试集
    train_size = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 训练随机森林
    rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS,
                               random_state=RANDOM_SEED,
                               n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # 创建当前币种独立文件夹
    curr_dir = os.path.join(RESULT_ROOT, currency_name)
    os.makedirs(curr_dir, exist_ok=True)

    # 1. 保存评估指标文本
    txt_path = os.path.join(curr_dir, f"{currency_name}_随机森林结果.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"币种：{currency_name}\n")
        f.write("===== 随机森林回归 模型评估指标 =====\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")

    # 2. 保存预测明细CSV
    csv_path = os.path.join(curr_dir, f"{currency_name}_RF预测明细.csv")
    result_df = pd.DataFrame({
        "日期": y_test.index,
        "真实汇率值": y_test.values,
        "随机森林预测值": y_pred
    })
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 3. 绘制并保存预测图
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, y_test.values, label="真实汇率值", color="#1f77b4", linewidth=1.2)
    plt.plot(y_test.index, y_pred, label="随机森林预测值", color="#9467bd",
             linewidth=1.2, linestyle="--")
    plt.title(f"随机森林模型 {currency_name}兑人民币汇率 预测结果对比")
    plt.legend()
    plt.grid(alpha=0.3)
    img_path = os.path.join(curr_dir, f"{currency_name}_RF预测图.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ {currency_name} 随机森林建模完成，结果已保存")

def main():
    # 读取数据
    df = pd.read_csv(DATA_PATH, index_col="日期", parse_dates=["日期"])
    df = df.replace("-", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")

    # 遍历所有币种
    currency_list = df.columns.tolist()
    print("===== 开始全币种随机森林回归预测 =====")
    for curr in currency_list:
        print(f"\n---------- 正在处理：{curr} ----------")
        run_rf_single_currency(curr, df[curr])

    print("\n===== 全币种随机森林预测全部完成 =====")

if __name__ == "__main__":
    main()