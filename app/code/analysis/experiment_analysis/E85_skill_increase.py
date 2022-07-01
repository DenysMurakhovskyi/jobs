# -*- coding: utf-8 -*-
import sys

sys.path.append("/code")
import matplotlib.pyplot as plt
import pandas as pd
from app.code.CONFIG import *
from app.code.Tools import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # 设置figure_size尺


def read_dict(mode):
    df_dct = pd.read_csv("%s/%s/translate_wordcloud.csv" % (DATA_PATH, mode))
    trans_dct = {}
    for key, value in df_dct.values.tolist():
        trans_dct[key] = value
    return trans_dct


def skill_ratio(df_skill, high, out_path):
    draw_list = ["矩阵计算", "词性分析", "信息论", "计算语言学", "声纹识别", "plsa", "xgb"]
    df_now = df_skill[['jd_id', 'skill_name', 'value', 'oatt', 'high']]
    df_now = df_now[(df_now['high'] == high)].drop('high', axis=1)
    df_now['value_add'] = df_now[['value', 'oatt']].apply(lambda x: x[0] * x[1], axis=1)
    df_jd = df_now[['jd_id', 'value_add']]
    df_jd = df_jd.groupby('jd_id').sum().reset_index()
    df_jd.rename(columns={'value_add': 'salary'}, inplace=True)
    df_now = pd.merge(df_now, df_jd, on='jd_id')
    # df_now = df_now[df_now['skill_name'].isin(draw_list)]
    df_now['drop_sal'] = df_now[['salary', 'value_add', 'oatt']].apply(lambda x: (x[0] - x[1]) / (1.0 - x[2]), axis=1)
    df_now = df_now[['skill_name', 'drop_sal', 'salary', 'value', 'oatt', 'value_add']]
    df_now['ratio'] = df_now.apply(lambda x: (x[2] - x[1]) / x[1] if x[1] != 0 else 0, axis=1)
    df_now = df_now[['skill_name', 'ratio', 'salary', 'value', 'oatt', 'value_add']]
    df_now['skill_name'] = df_now['skill_name'].apply(lambda x: trans_dct[x])
    df_now.to_csv("%s/skill_increase_source.csv" % out_path, index=False)

    df_now = df_now.groupby('skill_name').mean().reset_index().sort_values(by='ratio', ascending=False)
    df_now.to_csv("%s/skill_increase.csv" % out_path, index=False)
    return


if __name__ == "__main__":
    mode = "IT"
    out_path = "%s/%s" % (OUT_PATH, mode)
    make_path(out_path)
    trans_dct = read_dict(mode)
    df_skill = pd.read_csv("%s/%s/value_save/value_test.csv" % (SAVE_PATH, mode))
    df_skill = df_skill.append(pd.read_csv("%s/%s/value_save/value_train.csv" % (SAVE_PATH, mode)))

    skill_ratio(df_skill, 0, out_path)
    df_ratio = pd.read_csv("%s/skill_increase.csv" % out_path)
    print(df_ratio)
