import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 配置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据、清洗脏数据
df = pd.read_csv("C:/Users/21182/Desktop/exchange-rate-forecast/data/processed_rmb_rate.csv", index_col="日期", parse_dates=["日期"])
df = df.replace("-", np.nan)
df = df.apply(pd.to_numeric, errors="coerce")  # 强制转数值，非法字符变NaN

# 有效数据阈值
MIN_VALID_LEN = 365
currency_list = df.columns.tolist()
print(f"本次待分析币种共 {len(currency_list)} 个：")
print(currency_list)
print("-" * 80)

# ADF检验函数：返回拼接好的结果字符串
def adf_test(ts, currency_name):
    result = adfuller(ts)
    adf_stat = result[0]
    p_val = result[1]
    crit = result[4]

    res_text = f"===== {currency_name} ADF平稳性检验结果 =====\n"
    res_text += f"ADF统计量: {adf_stat:.4f}\n"
    res_text += f"P值: {p_val:.4f}\n"
    res_text += "临界值：\n"
    for k, v in crit.items():
        res_text += f"  {k}: {v:.4f}\n"
    if p_val < 0.05:
        res_text += "✅ 结论：序列为平稳序列\n"
    else:
        res_text += "❌ 结论：序列为非平稳序列\n"
    print(res_text)
    return res_text

# 遍历币种
for currency in currency_list:
    try:
        print(f"\n========== 开始处理：{currency} ==========")
        series = df[currency].copy()
        valid_series = series.dropna()
        valid_count = len(valid_series)
        print(f"有效数据条数：{valid_count}")

        if valid_count < MIN_VALID_LEN:
            print(f"⚠️ {currency} 有效数据不足 {MIN_VALID_LEN} 条，跳过分析")
            plt.close()
            continue

        # 原始序列检验
        res1 = adf_test(valid_series, currency)
        # 一阶差分
        diff_series = valid_series.diff().dropna()
        print(f"\n===== {currency} 一阶差分后检验 =====")
        res2 = adf_test(diff_series, f"{currency}_一阶差分序列")

        # 合并所有结果
        total_text = f"币种名称：{currency}\n有效数据条数：{valid_count}\n\n" + res1 + "\n" + res2
        # 保存文本到result
        txt_path = f"C:/Users/21182/Desktop/exchange-rate-forecast/result/{currency}_分析结果.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(total_text)

        # 绘制并保存图片
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        valid_series.plot(label=f"原始{currency}", color="#1f77b4", linewidth=1)
        plt.title(f"{currency}兑人民币汇率（有效数据区间）")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(2, 1, 2)
        diff_series.plot(label=f"一阶差分序列", color="#ff7f0e")
        plt.title(f"{currency}汇率 一阶差分序列")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"C:/Users/21182/Desktop/exchange-rate-forecast/result/{currency}_station_test.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ {currency} 图表、文本结果已保存")

    except Exception as e:
        print(f"❌ {currency} 分析出错：{str(e)}")
        plt.close()
        continue

print("\n" + "="*80)
print("✅ 全部币种分析完成，结果已存入result文件夹")