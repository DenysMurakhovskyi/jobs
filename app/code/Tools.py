import pandas as pd
from .CONFIG import *
import os
from random import randint
from wordcloud import WordCloud
from scipy import stats
import numpy as np


def confidence_interval_mean95(data):
    return stats.norm.interval(alpha=0.95, loc=np.mean(data), scale=stats.sem(data))


def read_city_dct():
    df = pd.read_csv("%s/city_translate.csv" % DATA_PATH)
    dct = {}
    for key, val in df[['city_zh', 'city_en']].values.tolist():
        dct[key] = val
    return dct


def read_jd(mode):
    df = pd.read_csv("%s/%s/jd_expanded.csv" % (DATA_PATH, mode))
    df['job_title'] = df['job_title'].fillna('nan')
    df['job_title'] = df['job_title'].apply(lambda x: x.lower())
    return df


def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = randint(120, 250)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(randint(60, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)


def word_cloud(frequency_dict, out_path):
    wc = WordCloud(background_color='white',  # 背景颜色
                   max_words=100,  # 最大词数
                   max_font_size=100,  # 显示字体的最大值
                   random_state=30,
                   width=1600,
                   height=800,
                   color_func=random_color_func)
    wc.generate_from_frequencies(frequency_dict)
    wc.to_file(out_path)


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_low_high(salary):
    def get_salary(sal_str):
        num = -1
        for i,_ in enumerate(sal_str):
            try:
                num = float(sal_str[:i+1])
            except:
                break
        if num == -1: return num
        if salary.find('k') != -1:
            num = num * 1000
        return num / 1000.0
    low, high, mid = (-1, -1, -1)
    if salary.find('-') != -1:
        low, high = salary.split('-')
        low = get_salary(low)
        high = get_salary(high)
    elif salary.find(u'以上') != -1:
        low = get_salary(salary)
    elif salary.find(u'以下') != -1:
        high = get_salary(salary)
    if low != -1 and high != -1:
        mid = (low + high) / 2
    return low, mid, high
