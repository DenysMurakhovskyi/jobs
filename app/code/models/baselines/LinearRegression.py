import sys
sys.path.append("/code")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from CONFIG import DATA_PATH
from math import sqrt

def read_data(data_path, md=0):
    # ------------------- read information --------------
    with open(data_path, 'rb') as f:
        dct = pickle.load(f, encoding='utf-8')
    # dims
    n_level = dct['n_level']
    n_comp = dct['n_comp']
    n_city = dct['n_city']
    n_tempt = dct['n_tempt']
    n_workyear = dct['n_workyear']
    n_skill = dct['n_skill']
    n_industry = max([max(u[-1]) for u in dct['feat_comp']]) + 1
    
    # JD data
    time_ids = dct['time_index']
    salaries = dct['salary']
    ids = dct['id']
    skill_sets = dct['skill_set']
    comp_ids = dct['comp_id']
    city_ids = dct['city_id']
    work_years = dct['job_work_year']
    temptations = dct['job_temptation']

    # skill data
    data_skill_feat = dct['time_skill_feat']
    dim_skill = len(data_skill_feat[0][0])

    # company data
    feat_comp = dct['feat_comp']
    feat_time_city = dct['feat_time_city']
    feats, labels = [], []
    city_val_count = max([feat[2] for feat in feat_comp]) + 1
    len_of_sequences = list(map(lambda x: len(x), [ids, skill_sets, comp_ids,
                                                                                  city_ids, time_ids, salaries,
                                                                                  work_years, temptations]))
    pass

    for id, skill_set, comp_id, city_id, time_id, salary, work_year, tempt in zip(ids, skill_sets, comp_ids,
                                                                                  city_ids, time_ids, salaries,
                                                                                  work_years, temptations):
        # c = 0
        # if time_id == 0:
        #     continue
        skill_feat2 = [0] * dim_skill
        skill_feat = [0] * n_skill
        for i, level_skill in enumerate(skill_set):
            skill_feat[level_skill[1]] = level_skill[0] + 1
            skill_feat2 = [sk_f + sk_n / len(skill_set) for sk_f, sk_n in zip(skill_feat2, data_skill_feat[0][level_skill[1]])]

        data_tempt = [0] * n_tempt
        for tempt_id in tempt:
            data_tempt[tempt_id] = 1
        industry_fields = feat_comp[comp_id][-1]
        industry_feat = [0] * n_industry
        for field in industry_fields:
            industry_feat[field] = 1
        company_city_feat = [0] * city_val_count
        company_city_feat[feat_comp[comp_id][2]] = 1        

        feat = skill_feat + data_tempt + industry_feat + feat_comp[comp_id][:2] + feat_comp[comp_id][3:-1] + company_city_feat + feat_time_city[0][city_id] + [work_year, city_id, time_id] + skill_feat2
        if salary[md] != -1:
            feats.append(feat)
            labels.append(salary[md])
    return feats, labels

# 执行命令：python LinearRegression.py {dataset_name}
# 输出一次hold-out validation 结果


if __name__ == "__main__":
    # IT
    # dataset_name = "Designer" #sys.argv[1]
    dataset_name = 'IT'
    salary_bound = 'higher'
    md = 2 if salary_bound == 'higher' else 0
    
    path = "%s/%s/input_skillfeat.pkl" % (DATA_PATH, dataset_name)
    feats, labels = read_data(data_path=path, md=md)
    X_train, X_test, y_train, y_test = train_test_split(feats, labels, test_size=0.2)
    model = LinearRegression()
    model = model.fit(X_train, y_train)
    y_predict = model.predict(X_test).tolist()
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    print("md: %d"%md, "rmse: %f"%sqrt(mse), "mae: %f"%mae)
