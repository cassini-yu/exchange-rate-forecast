import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 全局配置
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 路径配置
RESULT_ROOT = "c:/Users/21182/Desktop/exchange-rate-forecast/result"
SAVE_DIR = "c:/Users/21182/Desktop/exchange-rate-forecast/result/summary"
os.makedirs(SAVE_DIR, exist_ok=True)

# 修正：更通用的正则，匹配单独的 MAE/MSE/RMSE
pattern_mae = re.compile(r"MAE[:：]\s*([\d\.]+)")
pattern_mse = re.compile(r"MSE[:：]\s*([\d\.]+)")
pattern_rmse = re.compile(r"RMSE[:：]\s*([\d\.]+)")

def extract_metrics(txt_path):
    """从文本文件提取 MAE、MSE、RMSE，提取失败返回空值"""
    if not os.path.exists(txt_path):
        return np.nan, np.nan, np.nan
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    mae_res = pattern_mae.search(content)
    mse_res = pattern_mse.search(content)
    rmse_res = pattern_rmse.search(content)

    mae = float(mae_res.group(1)) if mae_res else np.nan
    mse = float(mse_res.group(1)) if mse_res else np.nan
    rmse = float(rmse_res.group(1)) if rmse_res else np.nan
    return mae, mse, rmse

def main():
    all_currency = []
    arima_mae_list, arima_mse_list, arima_rmse_list = [], [], []
    holt_mae_list, holt_mse_list, holt_rmse_list = [], [], []
    rf_mae_list, rf_mse_list, rf_rmse_list = [], [], []

    # 遍历所有币种文件夹
    currency_dirs = [d for d in os.listdir(RESULT_ROOT) if os.path.isdir(os.path.join(RESULT_ROOT, d)) and d != "summary"]
    currency_dirs.sort()

    for curr in currency_dirs:
        curr_path = os.path.join(RESULT_ROOT, curr)
        all_currency.append(curr)

        # 1. 读取ARIMA指标：兼容两种可能的文件名
        arima_txt = None
        # 优先找最优结果文件，找不到再找普通结果文件
        if os.path.exists(os.path.join(curr_path, f"{curr}_ARIMA最优结果.txt")):
            arima_txt = os.path.join(curr_path, f"{curr}_ARIMA最优结果.txt")
        elif os.path.exists(os.path.join(curr_path, f"{curr}_ARIMA结果.txt")):
            arima_txt = os.path.join(curr_path, f"{curr}_ARIMA结果.txt")
        
        if arima_txt:
            a_mae, a_mse, a_rmse = extract_metrics(arima_txt)
        else:
            a_mae, a_mse, a_rmse = np.nan, np.nan, np.nan
        arima_mae_list.append(a_mae)
        arima_mse_list.append(a_mse)
        arima_rmse_list.append(a_rmse)

        # 2. 读取Holt指数平滑指标
        holt_txt = os.path.join(curr_path, f"{curr}_Holt指数平滑结果.txt")
        h_mae, h_mse, h_rmse = extract_metrics(holt_txt)
        holt_mae_list.append(h_mae)
        holt_mse_list.append(h_mse)
        holt_rmse_list.append(h_rmse)

        # 3. 读取随机森林指标
        rf_txt = os.path.join(curr_path, f"{curr}_随机森林结果.txt")
        r_mae, r_mse, r_rmse = extract_metrics(rf_txt)
        rf_mae_list.append(r_mae)
        rf_mse_list.append(r_mse)
        rf_rmse_list.append(r_rmse)

    # 构建汇总表格
    summary_df = pd.DataFrame({
        "币种": all_currency,
        "ARIMA_MAE": arima_mae_list,
        "ARIMA_MSE": arima_mse_list,
        "ARIMA_RMSE": arima_rmse_list,
        "Holt_MAE": holt_mae_list,
        "Holt_MSE": holt_mse_list,
        "Holt_RMSE": holt_rmse_list,
        "RandomForest_MAE": rf_mae_list,
        "RandomForest_MSE": rf_mse_list,
        "RandomForest_RMSE": rf_rmse_list
    })

    # 保存汇总CSV
    csv_save_path = os.path.join(SAVE_DIR, "全币种多模型指标汇总表.csv")
    summary_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 指标汇总表已保存：{csv_save_path}")

    # 绘制 RMSE 对比柱状图（核心评价指标）
    plt.figure(figsize=(18, 8))
    x = np.arange(len(all_currency))
    width = 0.25

    plt.bar(x - width, arima_rmse_list, width, label="ARIMA", color="#1f77b4")
    plt.bar(x, holt_rmse_list, width, label="Holt指数平滑", color="#2ca02c")
    plt.bar(x + width, rf_rmse_list, width, label="随机森林", color="#9467bd")

    plt.xticks(x, all_currency, rotation=45, ha="right")
    plt.xlabel("币种")
    plt.ylabel("RMSE（根均方误差）")
    plt.title("全币种三类模型 RMSE 精度对比")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    img_save_path = os.path.join(SAVE_DIR, "多模型RMSE对比图.png")
    plt.savefig(img_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 模型对比图表已保存：{img_save_path}")

    print("\n===== 指标汇总 & 绘图全部完成 =====")

if __name__ == "__main__":
    main()