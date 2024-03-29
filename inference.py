# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import argparse

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import optuna

# Custom library
from utils import seed_everything, print_score
from features import generate_label, feature_engineering1, feature_engineering2, plot_feature_importances


TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정


data_dir = '../input' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']
output_dir = '../output' # os.environ['SM_OUTPUT_DATA_DIR']

'''
    머신러닝 모델 없이 입력인자으로 받는 year_month의 이전 달 총 구매액을 구매 확률로 예측하는 베이스라인 모델
'''
def baseline_no_ml(df, year_month, total_thres=TOTAL_THRES):
    # year_month에 해당하는 label 데이터 생성
    month = generate_label(df, year_month)
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_d = d - dateutil.relativedelta.relativedelta(months=1)
    prev_d = prev_d.strftime('%Y-%m')
    
    # 이전 월에 해당하는 label 데이터 생성
    previous_month = generate_label(df, prev_d)
    
    # merge하기 위해 컬럼명 변경
    previous_month = previous_month.rename(columns = {'total': 'previous_total'})

    month = month.merge(previous_month[['customer_id', 'previous_total']], on = 'customer_id', how = 'left')
    
    # 거래내역이 없는 고객의 구매액을 0으로 채움
    month['previous_total'] = month['previous_total'].fillna(0)
    # 이전 월의 총 구매액을 구매액 임계값으로 나눠서 예측 확률로 계산
    month['probability'] = month['previous_total'] / total_thres
    
    # 이전 월 총 구매액이 구매액 임계값을 넘어서 1보다 클 경우 예측 확률을 1로 변경
    month.loc[month['probability'] > 1, 'probability'] = 1
    
    # 이전 월 총 구매액이 마이너스(주문 환불)일 경우 예측 확률을 0으로 변경
    month.loc[month['probability'] < 0, 'probability'] = 0
    
    return month['probability']


def make_lgb_prediction(train, y, test, features, categorical_features='auto', model_params=None):
    x_train = train[features]
    x_test = test[features]
    
    print(x_train.shape, x_test.shape)

    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # LightGBM 데이터셋 선언
    dtrain = lgb.Dataset(x_train, label=y)

    # LightGBM 모델 훈련
    clf = lgb.train(
        model_params,
        dtrain,
        categorical_feature=categorical_features,
        verbose_eval=200
    )
    
    # 테스트 데이터 예측
    test_preds = clf.predict(x_test)

    # 피처 중요도 저장
    fi['importance'] = clf.feature_importance()
    
    return test_preds, fi


def make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10):
    # #############MLFLOW##################
    # import mlflow
    # HOST = "http://localhost"
    # mlflow.set_tracking_uri(HOST+":6006/")
    # mlflow.start_run()
    # #############MLFLOW##################

    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)
        
        # LightGBM 모델 훈련
        clf = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid], # Validation 성능을 측정할 수 있도록 설정
            categorical_feature=categorical_features,
            verbose_eval=200
        )

        # Validation 데이터 예측
        val_preds = clf.predict(x_val)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test) / folds
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력

    # #############MLFLOW##################
    # mlflow.log_param("folds", folds)
    # for k, v in model_params.items():
    #     mlflow.log_param(k, v)
    #
    # mlflow.log_metric("Mean AUC", score)
    # mlflow.log_metric("OOF AUC", roc_auc_score(y, y_oof))
    # mlflow.end_run()
    # #############MLFLOW##################

    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi


def make_xgb_oof_prediction(train, y, test, features, model_params=None, folds=10):
    x_train = train[features]
    x_test = test[features]

    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])

    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])

    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0

    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features

    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]

        print(f'fold: {fold + 1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # XGBoost 데이터셋 선언
        dtrain = xgb.DMatrix(x_tr, label=y_tr)
        dvalid = xgb.DMatrix(x_val, label=y_val)

        # XGBoost 모델 훈련
        clf = xgb.train(
            model_params,
            dtrain,
            num_boost_round=10000,  # 트리 개수
            evals=[(dtrain, 'train'), (dvalid, 'valid')],  # Validation 성능을 측정할 수 있도록 설정
            verbose_eval=200,
            early_stopping_rounds=100
        )

        # Validation 데이터 예측
        val_preds = clf.predict(dvalid)

        # Validation index에 예측값 저장
        y_oof[val_idx] = val_preds

        # 폴드별 Validation 스코어 출력
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-' * 80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds

        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(xgb.DMatrix(x_test)) / folds

        # 폴드별 피처 중요도 저장
        fi_tmp = pd.DataFrame.from_records([clf.get_score()]).T.reset_index()
        fi_tmp.columns = ['feature', f'fold_{fold + 1}']
        fi = pd.merge(fi, fi_tmp, on='feature')

        del x_tr, x_val, y_tr, y_val
        gc.collect()

    print(f"\nMean AUC = {score}")  # 폴드별 평균 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}")  # Out Of Fold Validation 스코어 출력

    # 폴드별 피처 중요도 평균값 계산해서 저장
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)

    return y_oof, test_preds, fi


def make_cat_oof_prediction(train, y, test, features, categorical_features=None, model_params=None, folds=10):
    x_train = train[features]
    x_test = test[features]

    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])

    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])

    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0

    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features

    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]

        print(f'fold: {fold + 1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # CatBoost 모델 훈련
        clf = CatBoostClassifier(**model_params)
        clf.fit(x_tr, y_tr,
                eval_set=(x_val, y_val),  # Validation 성능을 측정할 수 있도록 설정
                cat_features=categorical_features,
                use_best_model=True,
                verbose=True)

        # Validation 데이터 예측
        val_preds = clf.predict_proba(x_val)[:, 1]

        # Validation index에 예측값 저장
        y_oof[val_idx] = val_preds

        # 폴드별 Validation 스코어 출력
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-' * 80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds

        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict_proba(x_test)[:, 1] / folds

        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold + 1}'] = clf.feature_importances_

        del x_tr, x_val, y_tr, y_val
        gc.collect()

    print(f"\nMean AUC = {score}")  # 폴드별 평균 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}")  # Out Of Fold Validation 스코어 출력

    # 폴드별 피처 중요도 평균값 계산해서 저장
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)

    return y_oof, test_preds, fi


# def objective(trial, label=label_2011_11):
#     lgb_params = {
#         'objective': 'binary',
#         'boosting_type': 'gbdt',
#         'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#         'max_depth': -1,
#         'max_bin': trial.suggest_int('max_bin', 128, 256),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 40),
#         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#         'n_estimators': 10000,
#         'early_stopping_rounds': 100,
#         'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#         'seed': SEED,
#         'verbose': -1,
#         'n_jobs': -1,
#     }


if __name__ == '__main__':

    # 인자 파서 선언
    parser = argparse.ArgumentParser()
    
    # baseline 모델 이름 인자로 받아서 model 변수에 저장
    parser.add_argument('model', type=str, default='baseline1', help="set baseline model name among baselin1,basline2,baseline3")
    args = parser.parse_args()
    model = args.model
    print('baseline model:', model)
    
    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])
    # data = pd.read_csv('/opt/ml/code/custom_dataset.csv', parse_dates=['year_month'])
    # 예측할 연월 설정
    year_month = '2011-12'
    
    if model == 'baseline1': # baseline 모델 1
        test_preds = baseline_no_ml(data, year_month)
    elif model == 'baseline2': # baseline 모델 2
        model_params = {
            'objective': 'binary', # 이진 분류
            'boosting_type': 'gbdt',
            'metric': 'auc', # 평가 지표 설정
            'feature_fraction': 0.8, # 피처 샘플링 비율
            'bagging_fraction': 0.8, # 데이터 샘플링 비율
            'bagging_freq': 1,
            'n_estimators': 100, # 트리 개수
            'seed': SEED,
            'verbose': -1,
            'n_jobs': -1,    
        }

        # 피처 엔지니어링 실행
        train, test, y, features = feature_engineering2(data, year_month)
        
        # LightGBM 모델 훈련 및 예측
        test_preds, fi = make_lgb_prediction(train, y, test, features, model_params=model_params)
    elif model == 'baseline3': # baseline 모델 3
        model_params = {
            'objective': 'binary', # 이진 분류
            'boosting_type': 'gbdt',
            'metric': 'auc', # 평가 지표 설정
            'feature_fraction': 0.8, # 피처 샘플링 비율
            'bagging_fraction': 0.8, # 데이터 샘플링 비율
            'bagging_freq': 1,
            'n_estimators': 10000, # 트리 개수
            'early_stopping_rounds': 100,
            'seed': SEED,
            'verbose': -1,
            'n_jobs': -1,    
        }
        
        # 피처 엔지니어링 실행
        train, test, y, features = feature_engineering2(data, year_month)
        # for x in train :
            # print("CHECK:",x)
        
        # Cross Validation Out Of Fold로 LightGBM 모델 훈련 및 예측
        # print("TEST:",features)
        print("`````")
        print(train)
        print(test)
        print(y)
        print(features)
        print("`````")
        y_oof, test_preds, fi = make_lgb_oof_prediction(train, y, test, features, model_params=model_params)
    elif model == 'baseline4':  # baseline 모델 3
        xgb_params = {
            'objective': 'binary:logistic',
            'learning_rate': 0.1,
            'max_depth': 6,
            'colsample_bytree': 0.8, # 피처 샘플링 80%
            'subsample': 0.8, # 데이터 샘플링 80%
            'eval_metric': 'auc',
            'seed': SEED,
        }

        # 피처 엔지니어링 실행
        train, test, y, features = feature_engineering2(data, year_month)
        # for x in train :
        # print("CHECK:",x)

        # Cross Validation Out Of Fold로 LightGBM 모델 훈련 및 예측
        # print("TEST:",features)
        y_oof, test_preds, fi = make_xgb_oof_prediction(train, y, test, features, model_params=xgb_params)
    elif model == 'baseline5':
        cat_params = {
            'n_estimators': 30000,
            'learning_rate': 0.07,
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'random_seed': SEED,
            'metric_period': 100,
            'od_wait': 100,
            'depth': 6,
            'rsm': 0.8
        }

        # 피처 엔지니어링 실행
        train, test, y, features = feature_engineering2(data, year_month)
        # for x in train :
        #     print("CHECK:",x)

        # Cross Validation Out Of Fold로 LightGBM 모델 훈련 및 예측
        print("TEST:",features)
        y_oof, test_preds, fi = make_cat_oof_prediction(train, y, test, features, model_params=cat_params)

    else:
        test_preds = baseline_no_ml(data, year_month)
    
    # 테스트 결과 제출 파일 읽기
    sub = pd.read_csv(data_dir + '/sample_submission.csv')
    
    # 테스트 예측 결과 저장
    # print("어케생겼누",test_preds)
    sub['probability'] = test_preds
    print(fi[['feature','importance']].sort_values(by='importance').values)
    # plot_feature_importances(fi)
    
    os.makedirs(output_dir, exist_ok=True)
    # 제출 파일 쓰기
    sub.to_csv(os.path.join(output_dir , 'output.csv'), index=False)