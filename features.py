import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
from matplotlib import pyplot as plt

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Custom library
from utils import seed_everything, print_score


TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정

data_dir = '../input/train.csv' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']


'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''
def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    df = df.copy()
    # year_month에 해당하는 label 데이터 생성
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    # df['year_month'] = df['year_month'].dt.strftime('%Y-%m')
    df.reset_index(drop=True, inplace=True)

    # year_month 이전 월의 고객 ID 추출
    cust = df[df['year_month']<year_month]['customer_id'].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df['year_month']==year_month]
    
    # label 데이터프레임 생성
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()

    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    # 고객 ID로 정렬
    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    return label


def feature_preprocessing(train, test, features, do_imputing=True):
    print("debug:",features)
    # for x in features :
        # print("DEBUG!!:",x)
    x_tr = train.copy()
    x_te = test.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            cate_cols.append(f)
            le = LabelEncoder()
            # train + test 데이터를 합쳐서 레이블 인코딩 함수에 fit
            le.fit(list(x_tr[f].values) + list(x_te[f].values))
            
            # train 데이터 레이블 인코딩 변환 수행
            x_tr[f] = le.transform(list(x_tr[f].values))
            
            # test 데이터 레이블 인코딩 변환 수행
            x_te[f] = le.transform(list(x_te[f].values))

    print('categorical feature:', cate_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기
        imputer = SimpleImputer(strategy='median')

        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])
    
    return x_tr, x_te


def feature_engineering1(df, year_month):
    df = df.copy()
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    # train = df[df['year_month'] < prev_ym]
    # test = df[df['year_month'] < year_month]

    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    # group by aggregation 함수 선언
    agg_func = ['mean','max','min','sum','count','std','skew']
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_func)
        # train_agg = train.loc[train['year_month'] < tr_ym].groupby(['customer_id']).agg(agg_func)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for col in train_agg.columns.levels[0]:
            for stat in train_agg.columns.levels[1]:
                new_cols.append(f'{col}-{stat}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns

    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id']).agg(agg_func)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    # print("DEBUG!:",features)
    return x_tr, x_te, all_train_data['label'], features

def change_product_id(df):
    df = df.copy()
    test = df[df['product_id'].str.isdigit()==False]
    result = test[test['product_id'].str.isalpha()==False]
    for_merge = result['product_id'].str[:-1]
    for_merge = pd.DataFrame(for_merge)
    new_df = df.merge(for_merge, left_index=True, right_index=True, how='left')
    new_df['product_id_y'] = new_df['product_id_y'].fillna(new_df['product_id_x'])
    new_df.drop(['product_id_x'],axis=1,inplace=True)
    new_df.rename(columns = {'product_id_y':'product_id'},inplace=True)
    return new_df


def add_POST(df):
    df = df.copy()
    df = df[['customer_id', 'product_id', 'total']]
    df[df['product_id'] == 'POST']
    df[df['total']<0] = 0
    df_grpd = df.groupby(['customer_id'], as_index=False).sum()[['customer_id', 'total']]
    return df_grpd

def add_country(df):
    df = df.copy()
#     df = df[['customer_id', 'country']]
    i = 0
    countries = list(df['country'].unique())
    country_code = {}
    for country in countries :
        country_code[country] = i
        i+= 1
    country_series = pd.Series(country_code, name = 'country_code')
    country_df = pd.DataFrame(country_series, columns=['country','country_code'])
    country_df['country'] = country_series.index
    result = df.merge(country_df, on=['country'], how='left')
#     result = result[['customer_id', 'country_code']]
    return result

def add_eu(df):
    df = df.copy()
    EU_dict = {'Norway': 0, 'Switzerland': 0, 'Iceland': 0, 'USA': 0, 'Australia': 0, 'United Arab Emirates': 0, 'Japan': 0, 'Unspecified': 0, 'Nigeria': 0, 'RSA': 0, 'Singapore': 0, 'Bahrain': 0, 'Thailand': 0, 'Israel': 0, 'West Indies': 0, 'Korea': 0, 'Brazil': 0, 'Canada': 0, 'Lebanon': 0, 'Saudi Arabia': 0,
    'United Kingdom': 1, 'France': 1, 'Belgium': 1, 'EIRE': 1, 'Germany': 1, 'Portugal': 1, 'Denmark': 1, 'Netherlands': 1, 'Poland': 1, 'Channel Islands': 1, 'Spain': 1, 'Cyprus': 1, 'Greece': 1, 'Austria': 1, 'Sweden': 1, 'Finland': 1, 'Italy': 1, 'Malta': 1, 'Lithuania': 1, 'Czech Republic': 1, 'European Community': 1
    }
    eu_df = pd.DataFrame(pd.Series(EU_dict))
    eu_df['country'] = eu_df.index
    eu_df = eu_df[['country',0]]
    eu_df.columns=['country','eu']
    result = df.merge(eu_df, on=['country'], how='left')
    result = result[['customer_id','eu']]
    return result


def add_outlier(df):
    df = df.copy()

    product_id_outlier = 0
    product_id_outlier_set = set()

    for x in df['product_id'].values:
        if x[0:5].isnumeric() == True or (x[0:5].isnumeric() == True and x[-1].isalpha()):
            continue
        else:
            product_id_outlier_set.add(x)

    suspect_set = set()
    for outlier in product_id_outlier_set:
        tmp_list = df[df['product_id'] == outlier]['customer_id'].values
        for suspect in tmp_list:
            suspect_set.add(suspect)
    suspect_list = sorted(list(suspect_set))

    result_set = set()

    for sus in suspect_list:
        if len(df[df['customer_id'] == sus].product_id.values) < 20:
            for x in sorted(df[df['customer_id'] == sus].product_id.values):
                if x in product_id_outlier_set:
                    result_set.add(sus)
                else:
                    break

    result_dict = dict.fromkeys(result_set, 1)
    result_df = pd.DataFrame.from_dict(result_dict, orient='index', columns=['outlier'])
    result_df['customer_id'] = result_df.index
    real_df = df.merge(result_df, on=['customer_id'], how='left')
    return real_df

def feature_engineering2(df, year_month):
    df = df.copy()
    df['outlier'] = add_outlier(df)['outlier']
    df.fillna(0)
    df['product_id'] = change_product_id(df)['product_id']
    df['product_id'] = df['product_id'].str[:3]
    df['post'] = add_POST(df)['total']
    df['country_code'] = add_country(df)['country_code']
    df['eu'] = add_eu(df)['eu']
    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()
    df['cumsum_post_by_cust_id'] = df.groupby(['customer_id'])['post'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()

    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()

    # time series diff
    df['order_ts'] = df['order_date'].astype(np.int64) // 1e9
    df['order_ts_diff'] = df.groupby(['customer_id'])['order_ts'].diff()
    df['quantity_diff'] = df.groupby(['customer_id'])['quantity'].diff()
    df['price_diff'] = df.groupby(['customer_id'])['price'].diff()
    df['total_diff'] = df.groupby(['customer_id'])['total'].diff()

    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')

    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]

    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id', 'year_month', 'label']]
    test_label = generate_label(df, year_month)[['customer_id', 'year_month', 'label']]

    # group by aggregation 함수 선언
    agg_func = ['mean', 'max', 'min', 'sum', 'count', 'std', 'skew']
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
        'cumsum_total_by_cust_id': agg_func,
        'cumsum_quantity_by_cust_id': agg_func,
        'cumsum_price_by_cust_id': agg_func,
        'cumsum_post_by_cust_id': ['min','max','sum','std'],
        'cumsum_total_by_prod_id': agg_func,
        'cumsum_quantity_by_prod_id': agg_func,
        'cumsum_price_by_prod_id': agg_func,
        'cumsum_total_by_order_id': agg_func,
        'cumsum_quantity_by_order_id': agg_func,
        'cumsum_price_by_order_id': agg_func,
        'order_id': ['nunique'],
        'product_id': ['nunique'],
        'order_ts': ['first', 'last'],
        'order_ts_diff': agg_func,
        'quantity_diff': agg_func,
        'price_diff': agg_func,
        'total_diff': agg_func,
        'country_code':['last'],
        'eu':['last'],
        'post':['sum'],
        'outlier':['last'],
        # 'description':['nunique'],
    }
    all_train_data = pd.DataFrame()

    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)

        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                if type(stat) is str:
                    new_cols.append(f'{col}-{stat}')
                else:
                    new_cols.append(f'{col}-mode')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace=True)

        train_agg['year_month'] = tr_ym

        all_train_data = all_train_data.append(train_agg)

    # train_new = add_POST(train)
    # test_new = add_POST(test)

    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    # print("테스트",all_train_data.year_month)
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    # for x in features :
        # print("DDD:",x)

    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols

    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)

    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)

    return x_tr, x_te, all_train_data['label'], features


def plot_feature_importances(df, n=20, color='blue', figsize=(12, 8)):
    # 피처 중요도 순으로 내림차순 정렬
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)

    # 피처 중요도 정규화 및 누적 중요도 계산
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    plt.rcParams['font.size'] = 12
    plt.style.use('fivethirtyeight')
    # 피처 중요도 순으로 n개까지 바플롯으로 그리기
    df.loc[:n, :].plot.barh(y='importance_normalized',
                            x='feature', color=color,
                            edgecolor='k', figsize=figsize,
                            legend=False)

    plt.xlabel('Normalized Importance', size=18);
    plt.ylabel('');
    plt.title(f'Top {n} Most Important Features', size=18)
    plt.gca().invert_yaxis()
    plt.show()
    return df


if __name__ == '__main__':
    
    print('data_dir', data_dir)
