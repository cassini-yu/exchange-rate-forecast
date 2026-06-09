import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# 全局配置
warnings.filterwarnings("ignore")
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 路径配置
DATA_PATH = "C:/Users/21182/Desktop/exchange-rate-forecast/data/processed_rmb_rate.csv"
RESULT_ROOT = "C:/Users/21182/Desktop/exchange-rate-forecast/result"
# 阶数搜索范围 p,q
P_RANGE = [0, 1, 2, 3]
Q_RANGE = [0, 1, 2, 3]
D = 1  # 固定一阶差分

def train_select_best_arima(series, train_ratio=0.7):
    """
    单币种自动遍历阶数，选择最优ARIMA(p,1,q)
    :param series: 单币种时序数据
    :param train_ratio: 训练集比例
    :return: 最优模型、最优参数、最优指标、所有候选结果
    """
    series = series.dropna()
    n = len(series)
    split = int(n * train_ratio)
    train = series.iloc[:split]
    test = series.iloc[split:]

    best_aic = float("inf")
    best_bic = float("inf")
    best_rmse = float("inf")
    best_model = None
    best_order = None
    candidate_records = []

    for p in P_RANGE:
        for q in Q_RANGE:
            order = (p, D, q)
            try:
                model = ARIMA(train, order=order)
                res = model.fit()

                # 1. 检查系数显著性：所有系数P值<0.05
                p_vals = res.pvalues
                if (p_vals > 0.05).any():
                    continue

                # 2. 检查残差白噪声 Ljung-Box
                lb_p = res.test_serial_correlation("ljungbox", lags=1)[0][0][1]
                if lb_p < 0.05:
                    continue

                # 3. 预测并计算误差
                pred = res.get_prediction(start=split, end=n-1).predicted_mean
                mse = np.mean((pred - test) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(pred - test))

                aic = res.aic
                bic = res.bic

                candidate_records.append({
                    "order": order,
                    "aic": aic,
                    "bic": bic,
                    "rmse": rmse,
                    "mae": mae,
                    "lb_p": lb_p
                })

                # 更新最优模型：优先AIC最小
                if aic < best_aic:
                    best_aic = aic
                    best_bic = bic
                    best_rmse = rmse
                    best_model = res
                    best_order = order

            except Exception:
                # 跳过模型报错、不稳定的阶数
                continue

    return best_model, best_order, candidate_records

def main():
    # 读取数据 + 补全时间频率，消除索引警告
    df = pd.read_csv(DATA_PATH, index_col="日期", parse_dates=["日期"])
    df = df.replace("-", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.asfreq("D")  # 补全日频率
    currency_list = df.columns.tolist()

    for curr in currency_list:
        print(f"\n========== 正在优化币种：{curr} ==========")
        ts = df[curr]
        if len(ts.dropna()) < 300:
            print(f"数据量不足，跳过 {curr}")
            continue

        # 创建币种独立文件夹
        curr_dir = os.path.join(RESULT_ROOT, curr)
        os.makedirs(curr_dir, exist_ok=True)

        # 自动寻优
        best_res, best_ord, all_cand = train_select_best_arima(ts)
        if best_res is None:
            print(f"{curr} 未找到合适ARIMA阶数")
            continue

        p_opt, d_opt, q_opt = best_ord
        print(f"{curr} 最优阶数: ARIMA({p_opt},{d_opt},{q_opt})")

        # 划分训练/测试集
        split = int(len(ts.dropna()) * 0.7)
        train = ts.dropna().iloc[:split]
        test = ts.dropna().iloc[split:]
        pred = best_res.get_prediction(start=split, end=len(ts.dropna())-1).predicted_mean

        # 计算评价指标
        mae = np.mean(np.abs(pred - test))
        mse = np.mean((pred - test) ** 2)
        rmse = np.sqrt(mse)

        # 1. 保存最优模型信息、指标、所有候选阶数结果
        res_txt = os.path.join(curr_dir, f"{curr}_ARIMA最优结果.txt")
        with open(res_txt, "w", encoding="utf-8") as f:
            f.write(f"币种：{curr}\n")
            f.write(f"最优模型阶数：ARIMA({p_opt},{d_opt},{q_opt})\n\n")
            f.write("===== 预测评价指标 =====\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n\n")
            f.write("===== 最优模型详细报告 =====\n")
            f.write(str(best_res.summary()))
            f.write("\n\n===== 所有候选阶数对比 =====\n")
            for item in all_cand:
                f.write(f"order:{item['order']}, AIC:{item['aic']:.2f}, BIC:{item['bic']:.2f}, RMSE:{item['rmse']:.2f}\n")

        # 2. 绘制预测对比图
        plt.figure(figsize=(15, 6))
        plt.plot(train.index, train.values, label="训练集", color="#1f77b4")
        plt.plot(test.index, test.values, label="真实值", color="#2ca02c")
        plt.plot(test.index, pred.values, label="预测值", color="#ff7f0e", linestyle="--")
        plt.title(f"{curr} 汇率 ARIMA({p_opt},{d_opt},{q_opt}) 预测结果")
        plt.xlabel("时间")
        plt.ylabel("汇率")
        plt.legend()
        plt.grid(alpha=0.3)
        img_path = os.path.join(curr_dir, f"{curr}_ARIMA最优预测图.png")
        plt.tight_layout()
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ {curr} 最优模型结果已保存至：{curr_dir}")

    print("\n==================== 全币种ARIMA单独寻优完成 ====================")

if __name__ == "__main__":
    main()