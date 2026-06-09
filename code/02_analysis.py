import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

data_path = "C:/Users/21182/Desktop/exchange-rate-forecast/data/processed_rmb_rate.csv"
result_root = "C:/Users/21182/Desktop/exchange-rate-forecast/result" 

# 读取数据、清洗脏数据
df = pd.read_csv(data_path, index_col="日期", parse_dates=["日期"])
df = df.replace("-", np.nan)
df = df.apply(pd.to_numeric, errors="coerce")  # 强制转数值，非法字符变NaN

# 有效数据阈值
MIN_VALID_LEN = 365
currency_list = df.columns.tolist()
print(f"本次待分析币种共 {len(currency_list)} 个：")
print(currency_list)
print("-" * 80)


# 遍历币种
for currency in currency_list:
    print(f"\n========== 正在分析：{currency} ==========")
    series = df[currency].dropna()
    if len(series) < 365:
        print(f"⚠️ {currency} 数据量不足，跳过")
        continue

    # 1. 创建当前币种独立文件夹（和ARIMA目录统一）
    curr_dir = os.path.join(result_root, currency)
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)

    # 2. 基础统计描述，保存为txt
    desc_txt_path = os.path.join(curr_dir, f"{currency}_统计描述.txt")
    desc = series.describe()
    with open(desc_txt_path, "w", encoding="utf-8") as f:
        f.write(f"币种：{currency}\n")
        f.write(f"数据起止时间：{series.index[0]} ~ {series.index[-1]}\n")
        f.write(f"有效数据量：{len(series)}\n\n")
        f.write(str(desc))

    # 3. 绘制原始时序图并保存
    plt.figure(figsize=(14, 6))
    plt.plot(series.index, series.values)
    plt.title(f"{currency} 汇率原始时序图")
    plt.xlabel("时间")
    plt.ylabel("汇率")
    plt.grid(alpha=0.3)
    img1 = os.path.join(curr_dir, f"{currency}_时序图.png")
    plt.tight_layout()
    plt.savefig(img1, dpi=300, bbox_inches="tight")
    plt.close()

    # 4. 绘制一阶差分图
    diff_series = series.diff().dropna()
    plt.figure(figsize=(14, 6))
    plt.plot(diff_series.index, diff_series.values)
    plt.title(f"{currency} 汇率一阶差分序列图")
    plt.xlabel("时间")
    plt.ylabel("差分后汇率")
    plt.grid(alpha=0.3)
    img2 = os.path.join(curr_dir, f"{currency}_一阶差分图.png")
    plt.tight_layout()
    plt.savefig(img2, dpi=300, bbox_inches="tight")
    plt.close()

    # 5. ACF、PACF 图（定阶使用）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(diff_series, lags=30, ax=ax1)
    plot_pacf(diff_series, lags=30, ax=ax2, method="ywm")
    plt.suptitle(f"{currency} 差分序列 ACF/PACF 图")
    img3 = os.path.join(curr_dir, f"{currency}_ACF_PACF图.png")
    plt.tight_layout()
    plt.savefig(img3, dpi=300, bbox_inches="tight")
    plt.close()

    # 6. ADF平稳性检验结果保存
    adf_result = adfuller(diff_series)
    adf_txt_path = os.path.join(curr_dir, f"{currency}_ADF平稳性检验.txt")
    with open(adf_txt_path, "w", encoding="utf-8") as f:
        f.write(f"币种：{currency} 一阶差分序列 ADF检验\n")
        f.write(f"ADF统计量: {adf_result[0]:.4f}\n")
        f.write(f"P值: {adf_result[1]:.4f}\n")
        f.write(f"临界值:\n")
        for k, v in adf_result[4].items():
            f.write(f"  {k}: {v:.4f}\n")
        if adf_result[1] < 0.05:
            f.write("\n结论：一阶差分后序列平稳\n")
        else:
            f.write("\n结论：一阶差分后序列仍非平稳\n")

    print(f"✅ {currency} 分析结果已保存至：{curr_dir}")

print("\n==================== 全部币种数据分析完成 ====================")