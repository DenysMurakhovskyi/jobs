# -*- coding: utf-8 -*-
import sys
sys.path.append("../../../../jobs/code")
import pandas as pd
import seaborn as sns
from app.code.analysis.boxplot_info import boxplot_distribution
from pylab import *
from app.code.CONFIG import *
from app.code.Tools import *

translation_dict = {u'通用': 'Versatile', u'较强': 'Strong', u'良好': 'Well', u'能看懂': 'Can Read', u'精通': 'Specialist',
                    u'理解': 'Understand', u'熟悉': 'Familiar', u'灵活运用': 'Flexible Use', u'深刻理解': 'Profound',
                    u'擅长': 'Good At', u'掌握': 'Can Use', u"基本": 'Basic', u'关键': 'Crucial', u'优秀': 'Excellent',
                    u'了解': 'Know', u'丰富': 'Rich', u'架构':'Architecture', u'分布式系统': 'Distributed System',
                    u'推荐系统': 'Recommender system', u'c/c++': 'c/c++', u'golang': 'golang', u'应届毕业生':'graduate',
                    u'项目管理': 'project management', 'python':'Python', 'linux':'Linux','数据库': 'Database',
                    u'1年以下':'under 1', u'1-3':'1-3', u'3-5':'3-5', u'5-10':'5-10', u'10年以上':'above 10'}


def skill_value(df_skill):
    df_now = df_skill[['skill_name', 'value', 'high']]
    df_now = df_now.groupby(['skill_name', 'high']).mean().reset_index()
    df_now.rename(columns={'value': 'mean_value'}, inplace=True)
    df_now_high = df_now[df_now['high']==1].sort_values(by='mean_value', ascending=False).drop('high', axis=1)
    df_now_low = df_now[df_now['high']==0].sort_values(by='mean_value', ascending=False).drop('high', axis=1)
    # 前几名, 列表
    return df_now_high, df_now_low


def level_value_err(df_input, out_path, order_lst):
    plt.rcParams['figure.figsize'] = (8.0, 4.0)  # 设置figure_size尺
    df_now = df_input[['level_name', 'Influence', 'high']]
    df_now = df_now[df_now['level_name'].isin(order_lst)]
    df_now['high'] = df_now['high'].apply(lambda x: 'Lower-Bound' if x == 0 else 'Upper-Bound')
    df_now.rename(columns={'level_name': 'Level'}, inplace=True)
    df_now = df_now[df_now['Level'] != 'crucial']
    df_now = df_now[['Level', 'Influence', 'high']]

    plt.plot(order_lst, [0] * len(order_lst), linestyle='-.', linewidth=1.8, color='black', zorder=0)
    sns.pointplot(x="Level", y="Influence", hue="high", data=df_now, order=order_lst)

    #x1 = [-1 + i * 0.5 for i in range(5)]
    #x2 = ['%.1f' % (-100 + i * 50) for i in range(5)]
    plt.legend(fontsize=16, loc='upper right', ncol=2)
    #plt.yticks(x1, x2, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(rotation=40, fontsize=16)
    plt.xlabel("Level", fontsize=16)
    plt.ylabel("Influence(%)", fontsize=16)
    plt.savefig("%s/level_influence.png"%out_path)
    plt.close()


def pre_process(df_skill, out_path):
    df_skill_high, df_skill_low = skill_value(df_skill)
    df_skill_high.rename(columns={"mean_value": "high_value"}, inplace=True)
    df_skill_low.rename(columns={"mean_value": "low_value"}, inplace=True)
    df_skill_value = pd.merge(df_skill_high, df_skill_low, on='skill_name')  # skill, high_value, low_value

    df_now = df_skill[['skill_name', 'level_name', 'value', 'high']]
    df_now = pd.merge(df_now, df_skill_value, on='skill_name')  # skill, level, value, high, high_value, low_value
    df_now['value'] = df_now[['value', 'high_value', 'low_value', 'high']].apply(
        lambda x: (x[0] + 0.001) / (x[1] + 0.001) if x[-1] == 1 else (x[0] + 0.001) / (x[2] + 0.001), axis=1)
    df_now = df_now[['high', 'level_name', 'value']]
    df_now['level_name'] = df_now['level_name']  # .apply(lambda x: x.encode('utf-8').decode('utf-8'))
    df_now = df_now[df_now['level_name'] != u'其他']
    #df_now['level_name'] = df_now['level_name'].apply(lambda x: translation_dict[x])
    df_now['Influence'] = df_now['value'].apply(lambda x: (x - 1) * 100)
    df_now = df_now[['level_name', 'Influence', 'high']]
    df_now.to_csv("%s/%s_level.csv" % (out_path, dataset_name), index=False)

    return df_now


def boxplot_dis(df_input, out_path, order_lst):
    plt.rcParams['figure.figsize'] = (8.0, 4.0)  # 设置figure_size尺
    df_now = df_input[['level_name', 'Influence', 'high']]

    df_now = df_now[df_now['level_name'].isin(order_lst)]
    df_now['high'] = df_now['high'].apply(lambda x: 'Lower-Bound' if x == 0 else 'Upper-Bound')
    df_now.rename(columns={'level_name': 'Level'}, inplace=True)
    df_now = df_now[df_now['Level'] != 'crucial']
    df_now = df_now[['Level', 'Influence', 'high']]

    boxplot_distribution("%s/%s_level_statistics.csv"%(out_path, dataset_name), df=df_now, x='Level', y='Influence', hue='high')
    sns.boxplot(data=df_now, x='Level', y='Influence', hue='high', showfliers=False, order=order_lst) #  showmeans=True,

    #x1 = [-1 + i * 0.5 for i in range(5)]
    #x2 = ['%.1f' % (-100 + i * 50) for i in range(5)]
    plt.legend(fontsize=16, loc='upper right', ncol=2)
    #plt.yticks(x1, x2, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(rotation=60, fontsize=16)
    plt.xlabel("Level", fontsize=16)
    plt.ylabel("Influence(%)", fontsize=16)
    plt.savefig("%s/level_influence_boxplot.png"%out_path)
    plt.close()


def read_data(mode):
    df_jd = pd.read_csv("%s/%s/jd_expanded.csv"%(DATA_PATH, mode))[['id']]
    df_skill = pd.read_csv("%s/%s/value_save/value_test.csv"%(SAVE_PATH, mode))
    df_skill = df_skill.append(pd.read_csv("%s/%s/value_save/value_train.csv"%(SAVE_PATH, mode)))
    df_skill = pd.merge(df_skill, df_jd, left_on='jd_id', right_on='id').drop('jd_id', axis=1)
    drop_dct = {"IT": [], "Designer": []}
    order_dct = {"IT": ['know', 'canuse', 'familiar', 'experience', 'skill', 'strong', 'goodat', 'profound', 'effective', 'proficient', 'specialist', 'excellent', 'understand', 'profound', 'expertise', 'exceptional'],
                 "Designer": []}
    
    return df_skill, drop_dct, order_dct

# 运行命令：python E1_level_influence.py {dataset_name}
# 生成目录：out/{dataset_name}/
# 生成文件：
# bar_level.csv
# boxplot_level.csv
# boxplot_level_statistics.csv
# level_influence.png # Figure 2 & Figure S6
# level_influence_boxplot.png # Figure S8


if __name__=="__main__":
    dataset_name = "IT"
    #dataset_name = sys.argv[1]
    out_path = "%s/%s"%(OUT_PATH, dataset_name)
    make_path(out_path)
    df_skill, drop_dct, order_dct = read_data(dataset_name)
    df_now = pre_process(df_skill, out_path=out_path)
    print(df_skill)
    level_value_err(df_now, out_path=out_path, order_lst=order_dct[dataset_name])
    boxplot_dis(df_now, out_path=out_path, order_lst=order_dct[dataset_name])
