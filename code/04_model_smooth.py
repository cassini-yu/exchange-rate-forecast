import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 全局配置
warnings.filterwarnings("ignore")
plt.switch_backend("Agg")  # 后台绘图，不弹出窗口
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 路径配置（与02、03保持统一）
DATA_PATH = "C:/Users/21182/Desktop/exchange-rate-forecast/data/processed_rmb_rate.csv"
RESULT_ROOT = "C:/Users/21182/Desktop/exchange-rate-forecast/result"
TRAIN_RATIO = 0.8  # 训练集比例

def run_holt_single_currency(currency_name, ts_series):
    """单币种霍尔特线性指数平滑建模、预测、保存结果"""
    ts_clean = ts_series.dropna()
    if len(ts_clean) < 50:
        print(f"{currency_name} 有效数据过少，跳过")
        return

    # 划分训练集、测试集
    train_size = int(len(ts_clean) * TRAIN_RATIO)
    train = ts_clean.iloc[:train_size]
    test = ts_clean.iloc[train_size:]

    # 建模与训练
    model = Holt(train)
    model_fit = model.fit()

    # 预测
    predictions = model_fit.forecast(steps=len(test))

    # 计算评估指标
    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)

    # 创建当前币种结果文件夹
    curr_dir = os.path.join(RESULT_ROOT, currency_name)
    os.makedirs(curr_dir, exist_ok=True)

    # 1. 保存模型报告 + 评估指标文本
    txt_path = os.path.join(curr_dir, f"{currency_name}_Holt指数平滑结果.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"币种：{currency_name}\n")
        f.write("===== 霍尔特线性指数平滑 模型摘要 =====\n")
        f.write(str(model_fit.summary()))
        f.write("\n\n===== 模型评估指标 =====\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")

    # 2. 绘制并保存预测对比图
    plt.figure(figsize=(14, 6))
    plt.plot(test.index, test.values, label="真实汇率值", color="#1f77b4", linewidth=1.2)
    plt.plot(test.index, predictions.values, label="指数平滑预测值", color="#2ca02c",
             linewidth=1.2, linestyle="--")
    plt.title(f"霍尔特指数平滑 {currency_name}兑人民币汇率 预测结果对比")
    plt.legend()
    plt.grid(alpha=0.3)
    img_path = os.path.join(curr_dir, f"{currency_name}_Holt预测图.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ {currency_name} 霍尔特指数平滑完成，结果已保存")

def main():
    # 读取数据
    df = pd.read_csv(DATA_PATH, index_col="日期", parse_dates=["日期"])
    df = df.replace("-", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")

    # 遍历所有币种
    currency_list = df.columns.tolist()
    print("===== 开始全币种霍尔特线性指数平滑预测 =====")
    for curr in currency_list:
        print(f"\n---------- 正在处理：{curr} ----------")
        run_holt_single_currency(curr, df[curr])

    print("\n===== 全币种霍尔特指数平滑预测全部完成 =====")

if __name__ == "__main__":
    main()