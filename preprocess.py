import os, gc, sys
import re
import random
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

from googletrans import Translator

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import lightgbm as lgb
from catboost import CatBoost, Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, AdamW
import nlp

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed):
    """
    GPU+Pytorchを使用する場合の再現性確保のための関数.

    Parameters
    ----------
    seed: int
        固定するシードの値.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_googletranslate(params):
    """
    再翻訳してデータ増強を行うための関数.
    APIを使用する。リクエスト間隔は3秒間取っているため、実行時間が8時間ほど必要.
    
    Parameters
    ----------
    params: class(object)
        パラメータを保管したParametersクラス.
    """
    train = pd.read_csv(params.TRAIN_FILE)
    test = pd.read_csv(params.TEST_FILE)
    train_texts = train[params.TEXT_COL].tolist()
    test_texts = test[params.TEXT_COL].tolist()
 
    tslr = Translator()
    train_texts_en2fr2en = []
    for train_text in tqdm(train_texts):
        fr = tslr.translate(train_text, dest="fr")
        en = tslr.translate(fr.text, dest="en")
        train_texts_en2fr2en.append(en.text)
        time.sleep(3)
    train_de = pd.DataFrame({
        "id":train["id"].tolist(),
        params.TEXT_COL:train_texts_en2fr2en,
        params.TARGET:train[TARGET].tolist()
    })
    train_de.to_csv(params.BASE_PATH+"data/train_fr.csv", index=False)
 
    test_texts_en2fr2en = []
    for test_text in tqdm(test_texts):
        fr = tslr.translate(test_text, dest="fr")
        en = tslr.translate(fr.text, dest="en")
        test_texts_en2fr2en.append(en.text)
        time.sleep(3)
    test_de = pd.DataFrame({
        "id":test["id"].tolist(),
        params.TEXT_COL:test_texts_en2fr2en,
    })
    test_de.to_csv(params.BASE_PATH+"data/test_fr.csv", index=False)
 
    train_texts_en2de2en = []
    for train_text in tqdm(train_texts):
        de = tslr.translate(train_text, dest="de")
        en = tslr.translate(de.text, dest="en")
        train_texts_en2de2en.append(en.text)
        time.sleep(3)
    train_de = pd.DataFrame({
        "id":train["id"].tolist(),
        params.TEXT_COL:train_texts_en2de2en,
        params.TARGET:train[TARGET].tolist()
    })
    train_de.to_csv(params.BASE_PATH+"data/train_de.csv", index=False)
 
    test_texts_en2de2en = []
    for test_text in tqdm(test_texts):
        de = tslr.translate(test_text, dest="de")
        en = tslr.translate(de.text, dest="en")
        test_texts_en2de2en.append(en.text)
        time.sleep(3)
    test_de = pd.DataFrame({
        "id":test["id"].tolist(),
        params.TEXT_COL:test_texts_en2de2en,
    })
    test_de.to_csv(params.BASE_PATH+"data/test_de.csv", index=False)

    return train_fr, test_fr, train_de, test_de


def del_space(x):
    """
    クリーニング用の関数.
    余計な空白を除去する.
    
    Parameters:
    -----------
    x: str
        クリーニングしたいテキスト
    
    Returns:
    -----------
    x: str
        クリーニングしたテキスト
    """
    while '  ' in x:
        x = x.replace('  ', ' ')
    return x


def cleaning(texts):
    """from https://signate.jp/competitions/281/tutorials/17
    SGINATEチュートリアルから参照.
    データクリーニング用の関数.
    
    Parameters:
    -----------
    texts: List[str]
        クリーニングしたいテキストのリスト.

    Returns:
    -----------
    clean_texts: List[str]
        クリーニングしたテキストのリスト.
    """
    clean_texts = []
    stemmer = PorterStemmer()
    for text in texts:
        clean_punc = re.sub(r'[^a-zA-Z]', ' ', text)
        clean_short_tokenized = [word for word in clean_punc.split() if len(word) > 3]
        clean_normalize = [stemmer.stem(word) for word in clean_short_tokenized]
        clean_text = ' '.join(clean_normalize)
        clean_texts.append(clean_text)
    return clean_texts


def feature_extraction_vc(df, bottom_thld=0.0025, upper_thld=0.5):
    """
    Countベースのテキスト特徴量抽出関数.
    
    Parameters:
    -----------
    df: pd.DataFrame
        特徴量抽出をしたいデータフレーム.
    bottom_thld, upper_thld: float, float
        使用する特徴量の出現頻度の下限と上限.
        
    Returns:
    -----------
    voc_df: pd.DataFrame
        Countベースで特徴抽出したデータフレーム
    """
    vc = CountVectorizer()
    df = vc.fit_transform(df[params.TEXT_COL])
    voc_df = pd.DataFrame(df.toarray(), columns=vc.get_feature_names())
    use_cols = []
    for col in voc_df.columns:
        if voc_df.shape[0]*bottom_thld<voc_df[col].sum()<voc_df.shape[0]*upper_thld:
            use_cols.append(col)
    voc_df = voc_df[use_cols]
    voc_cols = {col:col+'_voc' for col in voc_df.columns}
    voc_df = voc_df.rename(columns=voc_cols)
    return voc_df


def feature_extraction_tfidf(df, bottom_thld=0.9):
    """
    tfidfベースのテキスト特徴量抽出関数.
    
    Parameters:
    -----------
    df: pd.DataFrame
        特徴量抽出をしたいデータフレーム.
    bottom_thld: float
        使用する特徴量の標準偏差の下限.

    Returns:
    -----------
    tdidf_df: pd.DataFrame
        tfidfベースで特徴抽出したデータフレーム
    """
    tfidf = TfidfVectorizer()
    df = tfidf.fit_transform(df)
    tfidf_df = pd.DataFrame(df.toarray(), columns=tfidf.get_feature_names())
    use_cols = []
    thld = np.percentile(tfidf_df.std().values, bottom_thld*100)
    for col in tfidf_df.columns:
        if thld < tfidf_df[col].std():
            use_cols.append(col)
    tfidf_df = tfidf_df[use_cols]
    tfidf_cols = {col:col+'_tfidf' for col in tfidf_df.columns}
    tfidf_df = tfidf_df.rename(columns=tfidf_cols)
    return tfidf_df


def preprocessing_lgb(vc_btm_thld=0.0025, vc_upr_thld=0.5, tfidf_thld=0.9):
    """
    lightgbm用の前処理関数.

    Parameters:
    ------------
    vc_btm_thld: float
        使用する特徴量の出現頻度の下限.
    vc_upr_thld: float
        使用する特徴量の出現頻度の上限.
    tfidf_thld: float
        tfidfで使用する特徴量の標準偏差の下限.

    Returns:
    ------------
    train: pd.DataFrame
        訓練用データフレーム.
    test: pd.DataFrame
        テスト用データフレーム.
    """
    train = pd.read_csv(params.TRAIN_FILE)
    test = pd.read_csv(params.TEST_FILE)
    test[params.TARGET] = -1

    df = pd.concat([train, test]).reset_index(drop=True)
    df[params.TEXT_COL] = df[params.TEXT_COL].apply(lambda x: del_space(x))
    
    df['description'] = cleaning(df['description'])
    voc_df = feature_extraction_vc(df, vc_btm_thld, vc_upr_thld)
    tfidf_df = feature_extraction_tfidf(df, tfidf_thld)
    df = pd.concat([pd.concat([train,test]).reset_index(drop=True), voc_df, tfidf_df], axis=1)
    train = df.iloc[:train.shape[0], :]
    test = df.iloc[train.shape[0]:, :]
    
    del voc_df, tfidf_df
    gc.collect()

    col = [c for c in train.columns if c not in ['id', params.TEXT_COL]]
    train = train[col]
    test = test[col].drop([params.TARGET], axis=1)
    return train, test


def get_train_data(params):
    """訓練用のデータフレームを返す.
    """
    return main(params)[0]

def get_test_data(params):
    """テスト用のデータフレームを返す.
    """
    return main(params)[1]

def main(params):
    train_fr, test_fr, train_de, test_de = get_googletranslate(params)

    lgb_train, lgb_test = preprocessing_lgb()
    lgb_train.to_csv(params.BASE_PATH+"data/lgb_train.csv", index=False)
    lgb_test.to_csv(params.BASE_PATH+"data/lgb_test.csv", index=False)

    train = pd.read_csv(params.TRAIN_FILE)
    test = pd.read_csv(params.TEST_FILE)
    return (train, train_fr, train_de, lgb_train), (test, test_fr, test_de, lgb_test)


class Parameters(object):
    """
    パラメータ管理用のクラス.
    """
    def __init__(self):
        self.SEED = 2020
        # コードのパス. os.getcwd()が動かない場合はstrで直接渡す.
        #BASE_PATH = "C:/StudentCup2020/"
        self.BASE_PATH = os.getcwd() + '/'
        self.TRAIN_FILE = self.BASE_PATH + "data/train.csv"
        self.TEST_FILE = self.BASE_PATH + "data/test.csv"
        self.TRAIN_FILE_FR = self.BASE_PATH+"data/train_fr.csv"
        self.TEST_FILE_FR = self.BASE_PATH+"data/test_fr.csv"
        self.TRAIN_FILE_DE = self.BASE_PATH+"data/train_de.csv"
        self.TEST_FILE_DE = self.BASE_PATH+"data/test_de.csv"
        self.TEXT_COL = "description"
        self.TARGET = "jobflag"
        self.NUM_CLASS = 4
params = Parameters()

if __name__ == "__main__":
    seed_everything(params.SEED)
    if "models" not in os.listdir(params.BASE_PATH):
        os.mkdir(params.BASE_PATH + "models/")
    main(params)