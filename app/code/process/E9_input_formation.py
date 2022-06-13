# -*- coding: utf-8 -*-
import pandas as pd
import sys
sys.path.append('../../../jobs/code/')
import pickle
from sklearn import preprocessing
import json
from app.code.Tools import get_low_high
from app.code.CONFIG import *
from app.code.Tools import *


def formulate_company(df_company):
    le = preprocessing.LabelEncoder()
    df_company['establish_year'] = df_company['establish_year'].astype(int)
    df_company['company_city'] = le.fit_transform(df_company['company_city'])
    print(df_company['company_city'].max())
    df_company['finance_stage'] = le.fit_transform(df_company['finance_stage'])
    fields = df_company['industry_field'].tolist()
    field_lst = []
    for field in fields:
        field_lst.extend(field)
    industry_le = preprocessing.LabelEncoder()
    industry_le.fit(field_lst)
    df_company['industry_field'] = df_company['industry_field'].apply(lambda x: industry_le.transform(x).tolist())
    return df_company, industry_le


def filter_jd(df_jd, n_company, n_city):
    # 取top city
    df_city_count = df_jd[['job_address']]
    df_city_count['count'] = 1
    df_city_count = df_city_count.groupby('job_address').sum()
    df_city_count['job_address'] = df_city_count.index
    df_city_count = df_city_count.sort_values(by='count', ascending=False).head(n_city)
    city_list = set(df_city_count['job_address'].tolist())
    df_jd = df_jd[df_jd['job_address'].isin(city_list)]

    # 取 top company
    df_company_count = df_jd[['comp_id']]
    df_company_count['count'] = 1
    df_company_count = df_company_count.groupby('comp_id').sum()
    df_company_count['comp_id'] = df_company_count.index
    #df_company_count = df_company_count.sort_values(by='count', ascending=False).head(n_company)
    df_company_count = df_company_count.sort_values(by='count', ascending=False)
    print("company_count", df_company_count.shape[0])
    company_list = set(df_company_count['comp_id'].tolist())
    df_jd = df_jd[df_jd['comp_id'].isin(company_list)]
    print("jd count", df_jd.shape[0])
    return df_jd, company_list, city_list


def city_input(city_list):
    # 读取城市特征并归一化
    # ['year', 'half_year','city', 'jd_count', 'low_p25','low_p50', 'low_p75', 'low_mean', 'low_std', 'mid_p25','mid_p50', 'mid_p75', 'mid_mean', 'mid_std', 'high_p25','high_p50', 'high_p75', 'high_mean', 'high_std']
    df_city = pd.read_csv("%s/city_features.csv"%DATA_PATH)
    df_city = df_city[df_city['city'].isin(city_list)]
    for col in df_city.columns[3:]:
        df_city[col] = preprocessing.scale(df_city[col])
    n_city = len(city_list)
    time_lst = []
    for year in [2021]:
        for half_year in [0]:
            time_lst.append([])
            for city in city_list:
                try:
                    city_feat = df_city[(df_city['city'] == city) & (df_city['year'] == year) & (df_city['half_year'] == half_year)].drop(['city', 'year', 'half_year'], axis=1).values.tolist()[0]
                except:
                    city_feat = [0] * (df_city.shape[1] - 3)
                time_lst[-1].append(city_feat)
    dct = {}
    for i, city in enumerate(city_list):
        dct[city] = i
    return time_lst, dct # n_time, n_city, dim_feat


def company_input(company_list, df_company):
    # 读取公司特征并转换Label
    # ['reg_capital', 'establish_year', 'company_city', 'finance_stage', 'industry_field']
    df_company = df_company[df_company['company_id'].isin(company_list)]
    df_company, industry_le = formulate_company(df_company)
    df_company['reg_capital'] = preprocessing.scale(df_company['reg_capital'])
    min_year = df_company['establish_year'].min()
    max_year = df_company['establish_year'].max()
    df_company['establish_year'] = df_company['establish_year'].apply(lambda x: (x * 1.0 - min_year) / (max_year - min_year))
    dct = {}
    for i, compid in enumerate(df_company['company_id'].tolist()):
        dct[compid] = i
    return df_company.drop('company_id', axis=1).values.tolist(), dct #


def input_skillfeat(dct):
    n_skill = dct['n_skill']
    n_time = dct['n_time']
    sal_lst = dct['salary']
    time_ids = dct['time_index']
    skill_sets = dct['skill_set']
    data = [[[] for i in range(n_time)] for j in ['low', 'high', 'mid', 'sub', 'skillcount']]
    data_count = [[(0, j) for j in range(n_skill)] for i in range(n_time)]
    for time, skill_set, sal in zip(time_ids, skill_sets, sal_lst):
        for skill, level in skill_set:
            if sal[0] != -1:
                data[0][time].append((sal[0], skill))
            if sal[2] != -1:
                data[1][time].append((sal[2], skill))
            if sal[0] != -1 and sal[2] != -1:
                data[2][time].append(((sal[0] + sal[1]) / 2.0, skill))
                data[3][time].append((sal[2] - sal[0] , skill))
            data[4][time].append((len(skill_set), skill))
            data_count[time].append((1, skill))
    time_ret = []
    for time in range(n_time):
        print(time)
        cols = ['count']
        df_count = pd.DataFrame(data_count[time], columns=['count', 'skill'])
        df_ret = df_count.groupby('skill').sum().reset_index() # skill, count
        for i, j in enumerate(['low', 'high', 'mid', 'sub']):
            df = pd.DataFrame(data[i][time], columns=['value', 'skill'])
            df_avg = df.groupby('skill').mean().reset_index()
            df_avg.rename(columns={'value': 'mean_%s'%j}, inplace=True)
            df_std = df.groupby('skill').std().reset_index()
            df_std.rename(columns={'value': 'std_%s'%j}, inplace=True)
            df_ret = pd.merge(df_ret, df_avg, on='skill', how='outer')
            df_ret = pd.merge(df_ret, df_std, on='skill', how='outer')
            cols.extend(['mean_%s'%j, 'std_%s'%j])
        df_ret = df_ret.fillna(0).sort_values(by='skill', ascending=True)
        assert (df_ret.shape[0] == n_skill), "df_ret must contain n_skill rows"
        time_ret.append(preprocessing.scale(df_ret[cols].values).tolist())
    dct['time_skill_feat'] = time_ret
    return dct

# 运行命令 python E9_input_formation.py {dataset_name}
# 输出目录 out/{dataset_name}/input_skillfeat.pkl


# dataset_name = "Designer"
if __name__ == "__main__":
    dataset_name = 'IT'
    with open("%s/%s/company_features.pkl"%(DATA_PATH, dataset_name), 'rb') as f:
        company_cols, company_data = pickle.load(f)
    df_company = pd.DataFrame(company_data, columns=company_cols)
    company_set = set(df_company['company_id'].tolist())

    # 读取skill和value
    with open("%s/%s/skill_level_dict.pkl"%(DATA_PATH, dataset_name), 'rb') as f:
        skill_dct, skill_ind_dct, level_dct, level_ind_dct, tempt_dct, tempt_ind_dct = pickle.load(f)
    n_skill = max(skill_ind_dct.values()) + 1
    n_level = max(level_ind_dct.values()) + 1
    n_tempt = 0 if len(tempt_ind_dct) == 0 else max(tempt_ind_dct.values()) + 1

    # 读取jd并预处理
    df_jd = pd.read_csv("%s/%s/jd_expanded.csv"%(DATA_PATH, dataset_name)) # time,comp_id,job_title,job_salary,job_address,skill_set
    df_jd = df_jd[df_jd['skill_set'].apply(lambda x: len(json.loads(x)) <= 40)]
    df_jd = df_jd[df_jd['comp_id'].isin(company_set)]    
    df_jd['year'] = df_jd['time'].apply(lambda x: int(x.split('/')[0].split('-')[0]))
    df_jd['half_year'] = df_jd['time'].apply(lambda x: (int(x.split('/')[0].split('-')[1])) / 6)
    
    df_jd, company_list, city_list = filter_jd(df_jd, n_company=1000, n_city=13)

    df_jd['skill_set'] = df_jd['skill_set'].apply(lambda x: sorted([(level_ind_dct[level_dct[dct['level']]], skill_ind_dct[skill_dct[dct['skill']]]) for dct in json.loads(x)], key=lambda x:x[1]))
    df_jd['job_salary'] = df_jd['job_salary'].apply(lambda x: get_low_high(x))
    df_jd = df_jd[df_jd['job_salary'].apply(lambda x: x[0] <= 200 and x[2] <= 300)]
    maxlen_skill = df_jd['skill_set'].apply(lambda x: len(x)).max()

    feat_time_city, city_ind_dict = city_input(city_list)
    feat_comp, comp_ind_dict = company_input(company_list, df_company)
    df_jd['job_address'] = df_jd['job_address'].apply(lambda x: city_ind_dict[x])
    df_jd['comp_id'] = df_jd['comp_id'].apply(lambda x: comp_ind_dict[x])
    df_jd['time'] = df_jd[['year', 'half_year']].apply(lambda x: int((x[0] - 2021) * 2 + x[1]), axis=1)
    df_jd = df_jd[(df_jd['time']>=0) & (df_jd['time']<=6)]
    df_jd['job_work_year'] = df_jd['job_work_year'].fillna('unlimited')
    df_jd['job_work_year'] = preprocessing.LabelEncoder().fit_transform(df_jd['job_work_year'])
    n_workyear = df_jd['job_work_year'].max() + 1
    df_jd['job_temptation'] = df_jd['job_temptation'].apply(lambda x: [tempt_ind_dct[tempt_dct[dct['temptation']]] for dct in json.loads(x)])

    ret_dct = {'n_workyear':n_workyear, 'n_time': 1, 'n_skill': n_skill, 'n_tempt':n_tempt, 'n_level': n_level,
               'n_comp': len(company_list), 'n_city': len(city_list), 'maxlen_skill': maxlen_skill,
               'time_index': df_jd['time'].tolist(), 'salary': df_jd['job_salary'].tolist(),
               'id': df_jd['id'].tolist(), 'skill_set': df_jd['skill_set'].tolist(),
               'comp_id': df_jd['comp_id'].tolist(), 'city_id': df_jd['job_address'].tolist(),
               'job_work_year': df_jd['job_work_year'].tolist(),
               'job_temptation': df_jd['job_temptation'].tolist(), 'feat_comp': feat_comp,
               'comp_ind_dict': comp_ind_dict, 'feat_time_city': feat_time_city, 'city_ind_dict': city_ind_dict}
    ret_dct = input_skillfeat(ret_dct)

    make_path("%s/%s"%(OUT_PATH, dataset_name))
    with open("%s/%s/input_skillfeat.pkl"%(OUT_PATH, dataset_name), 'wb') as f:
        pickle.dump(ret_dct, f)
