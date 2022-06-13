import numpy as np

def quantile25(x):
    return  np.quantile(a=x, q=0.25)

def quantile75(x):
    return  np.quantile(a=x, q=0.75)

def boxplot_distribution(path, df, x, y, hue=None):
    #df_tmp = df[df['Level'] == 'Can Use']
    #print(df_tmp)
    if hue == None:
        df_now = df[[x, y]].groupby(x).agg({y:['mean', quantile25, 'median', quantile75]})
    else:
        df_now = df[[x, hue, y]].groupby([x, hue]).agg({y:['mean', quantile25, 'median', quantile75]})
    df_now.to_csv(path)