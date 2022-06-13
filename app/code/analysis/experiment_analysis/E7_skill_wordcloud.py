# -*- coding: utf-8 -*-
import sys
sys.path.append("../../../../jobs/code")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from random import randint
from app.code.CONFIG import *
from app.code.Tools import random_color_func, make_path
plt.rcParams['figure.figsize'] = (16.0, 8.0)  # 设置figure_size尺


def read_dict(mode):
    df_dct = pd.read_csv("%s/%s/translate_wordcloud.csv"%(DATA_PATH, mode))
    trans_dct = {}
    for key,value in df_dct.values.tolist():
        trans_dct[key] = value
    return trans_dct


def skill_value(df_skill):
    df_now = df_skill[['skill_name', 'value', 'high']]
    df_now = df_now.groupby(['skill_name', 'high']).mean().reset_index()
    df_now.rename(columns={'value': 'mean_value'}, inplace=True)
    df_now_high = df_now[df_now['high']==1].sort_values(by='mean_value', ascending=False).drop('high', axis=1)
    df_now_low = df_now[df_now['high']==0].sort_values(by='mean_value', ascending=False).drop('high', axis=1)

    return df_now_high, df_now_low


def skill_domination(df_skill):
    df_now = df_skill[['skill_name', 'oatt', 'high']]
    df_now = df_now.groupby(['skill_name', 'high']).mean().reset_index()
    df_now.rename(columns={'oatt': 'mean_value'}, inplace=True)
    df_now_high = df_now[df_now['high']==1].sort_values(by='mean_value', ascending=False).drop('high', axis=1)
    df_now_low = df_now[df_now['high']==0].sort_values(by='mean_value', ascending=False).drop('high', axis=1)

    return df_now_high, df_now_low


def skill_contribution(df_skill):
    df_now = df_skill[['skill_name', 'value', 'oatt', 'high']]
    df_now['value'] = df_now[['value', 'oatt']].apply(lambda x: x[0] * x[1], axis=1)
    df_now = df_now.drop('oatt', axis=1)
    df_now = df_now.groupby(['skill_name', 'high']).mean().reset_index()
    df_now.rename(columns={'value': 'mean_value'}, inplace=True)
    df_now_high = df_now[df_now['high']==1].sort_values(by='mean_value', ascending=False).drop('high', axis=1)
    df_now_low = df_now[df_now['high']==0].sort_values(by='mean_value', ascending=False).drop('high', axis=1)

    return df_now_high, df_now_low


def read_data(mode):
    trans_dct = read_dict(mode)
    df_skill = pd.read_csv("%s/%s/value_save/value_test.csv" % (SAVE_PATH, mode))
    df_skill = df_skill.append(pd.read_csv("%s/%s/value_save/value_train.csv" % (SAVE_PATH, mode)))
    
    df_idf = pd.read_csv("%s/%s/skills_idf.csv"%(DATA_PATH, mode))
    return df_skill, trans_dct, df_idf


def df2dct(df_now):
    dict_val = {}
    for key, val in df_now.values.tolist():
        if key in trans_dct:
            if trans_dct[key] in dict_val:
                dict_val[trans_dct[key]] = max(dict_val[trans_dct[key]], val)
            else:
                dict_val[trans_dct[key]] = val
    return dict_val


def word_cloud(frequency_dict, out_path):
    wc = WordCloud(background_color='white',  # 背景颜色
                   max_words=100,  # 最大词数
                   max_font_size=100,  # 显示字体的最大值
                   random_state=30,
                   width=1600,
                   height=800,
                   color_func=random_color_func)

    wc.generate_from_frequencies(frequency_dict)
    
    lst = []
    for key, val in frequency_dict.items():
        lst.append((key, val))
    df = pd.DataFrame(lst, columns=['word', 'value'])
    df.to_csv(out_path + '.csv', index=False)
    wc.to_file(out_path + '.png')
    # plt.imshow(wc)
    # plt.axis('off')
    # plt.show()

# 执行命令：python E7_skill_wordcloud.py {dataset_name}
# 生成目录：out/{dataset_name}/
# 生成文件：
# value_wordcloud.csv
# value_wordcloud.png # Figure 4(a)
# domination_wordcloud.csv
# domination_wordcloud.png # Figure 4(b)
# contribution_wordcloud.csv
# contribution_wordcloud.png # Figure 4(c)


if __name__=="__main__":
    #mode = sys.argv[1] # "Designer"
    mode = "IT"
    make_path("%s/%s"%(OUT_PATH, mode))

    df_skill, trans_dct, idf_df = read_data(mode)
    df_skill = df_skill.merge(idf_df)
    df_skill['value'] = df_skill.value * df_skill.multiplier
    
    _, df_now_dom_low = skill_domination(df_skill)
    _, df_now_cont_low = skill_contribution(df_skill)
    _, df_now_value_low = skill_value(df_skill)

    dict_dm_val = df2dct(df_now_value_low)
    dict_dm_dom = df2dct(df_now_dom_low)
    dict_dm_cont = df2dct(df_now_cont_low)

    word_cloud(dict_dm_val, '%s/%s/%s_wordcloud_value'%(OUT_PATH, mode, mode))
    word_cloud(dict_dm_dom, '%s/%s/%s_wordcloud_domination'%(OUT_PATH, mode, mode))
    word_cloud(dict_dm_cont, '%s/%s/%s_wordcloud_contribution'%(OUT_PATH, mode, mode))
    
    df_skill['value'] = df_skill.value * df_skill.multiplier
    _, df_now_dom_low_idf = skill_domination(df_skill)
    _, df_now_cont_low_idf = skill_contribution(df_skill)
    _, df_now_value_low_idf = skill_value(df_skill)

    dict_dm_val_idf = df2dct(df_now_value_low_idf)
    dict_dm_dom_idf = df2dct(df_now_dom_low_idf)
    dict_dm_cont_idf = df2dct(df_now_cont_low_idf)

    word_cloud(dict_dm_val_idf, '%s/%s/%s_wordcloud_value_idf'%(OUT_PATH, mode, mode))
    word_cloud(dict_dm_dom_idf, '%s/%s/%s_wordcloud_domination_idf'%(OUT_PATH, mode, mode))
    word_cloud(dict_dm_cont_idf, '%s/%s/%s_wordcloud_contribution_idf'%(OUT_PATH, mode, mode))