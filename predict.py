import os, gc, sys
import re
import random
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import lightgbm as lgb
from catboost import CatBoost, Pool
import pulp

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


def np_rounder(x):
    """
    numpyの四捨五入用関数.

    Parameters:
    -----------
    x: np.array[float]

    Returns:
    ----------
    (int_array + float_array).astype(int): np.array[int]
    """
    int_array = x // 1
    float_array = x % 1
    float_array[float_array<0.5] = 0
    float_array[float_array>=0.5] = 1
    return (int_array + float_array).astype(int)

def sigmoid(x):
    """
    尤度を確率に変換する関数.

    Parameters:
    -----------
    x: np.array[float]

    Returns:
    1 / (1+np.exp(-x)) : np.array[float]
    """
    return 1 / (1+np.exp(-x))


def make_dataset(df, tokenizer, device, model_name):
    """
    NLPモデル用のデータセットを作成するための関数.

    Parameters:
    -----------
    df: pd.DataFrame
        モデル用のデータセット.
    tokenizer: transformers.AutoTokenizer.from_pretrained
        モデル用のtokenizer.
    device: str
        使用するデバイス. "cpu" or "cuda".
    model_name: str
        使用するモデルの名前.
    
    Returns:
    ----------
    dataset: nlp.Dataset.from_pandas
        NLP用のデータセット.
    """
    dataset = nlp.Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda example: tokenizer(example[params.TEXT_COL],
                                  padding="max_length",
                                  truncation=True,
                                  max_length=params.MAX_TOKEN_LEN))
    if not model_name in ["roberta-base", "distilbert-base-cased"]:
        dataset.set_format(type='torch', 
                           columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'], 
                           device=device)
    else:
        dataset.set_format(type='torch', 
                           columns=['input_ids', 'attention_mask', 'labels'], 
                           device=device)
    return dataset


def predict_lgb(X_test, n_folds=4):
    """
    lightgbm予測用関数.

    Parameters:
    -----------
    X_test: pd.DataFrame
        予測用データセット.
    n_folds: int
        予測時のFold数. 訓練時のFold数より大きくしないこと.
    
    Returns:
    ----------
    y_pred: np.array[float]
        予測した尤度.
    """
    y_pred = np.zeros((X_test.shape[0], params.NUM_CLASS), dtype='float32')
    for fold in range(n_folds):
        model = pickle.load(open(params.MODELS_DIR+"lgb_fold{}.lgbmodel".format(fold), "rb"))
        y_pred += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
    return y_pred


def predict_ctb(X_test, n_folds=4):
    """
    catboost予測用関数.

    Parameters:
    -----------
    X_test: pd.DataFrame
        予測用データセット.
    n_folds: int
        予測時のFold数. 訓練時のFold数より大きくしないこと.
    
    Returns:
    ----------
    y_pred: np.array[float]
        予測した尤度.
    """
    y_pred = np.zeros((X_test.shape[0], params.NUM_CLASS), dtype='float32')
    for fold in range(n_folds):
        model = pickle.load(open(params.MODELS_DIR+"ctb_fold{}.ctbmodel".format(fold), "rb"))
        y_pred += model.predict(X_test) / n_folds
    return y_pred


def predict_nlp(model_name, typ, file_path):
    """
    nlp予測用関数.

    Parameters:
    -----------
    model_name: str
        使用するモデルの名前.
    type: str
        使用する特徴量の部分.
    file_path: str
        予測するデータセットのパス.
    
    Returns:
    ----------
    preds: np.array[float]
        予測した尤度.
    """
    models = []
    for fold in range(params.NUM_SPLITS):
        model = Classifier(model_name, typ)
        model.load_state_dict(torch.load(params.MODELS_DIR + f"best_{model_name}_{typ}_{fold}.pth"))
        model.to(params.DEVICE)
        model.eval()
        models.append(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_df = pd.read_csv(file_path)
    test_df["labels"] = -1
    test_dataset = make_dataset(test_df, tokenizer, params.DEVICE, model_name)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params.VALID_BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        final_output = []
        preds = []
        for batch in test_dataloader:            
            if len(batch.values())==4:
                attention_mask, input_ids, labels, token_type_ids = batch.values()
            else:
                attention_mask, input_ids, labels = batch.values()
                token_type_ids = None
            pred = np.zeros((labels.shape[0], params.NUM_CLASS))
            for model in models:
                pred += model(input_ids, attention_mask, token_type_ids).cpu().numpy()
            preds += (pred/params.NUM_SPLITS).tolist()
    return preds


def hack(prob):
    """
    from: https://signate.jp/competitions/281/discussions/20200816040343-8180
    尤度最大化用関数.

    Parameters:
    ------------
    prob: np.array[float]
        予測した確率.
    
    Returns:
    ------------
    x_ast.argmax(axis=1): np.array[int]
        予測したラベル.
    """
    logp = np.log(prob + 1e-16)
    N = prob.shape[0]
    K = prob.shape[1]
    m = pulp.LpProblem('Problem', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', [(i, j) for i in range(N) for j in range(K)], 0, 1, pulp.LpBinary)
    log_likelihood = pulp.lpSum([x[(i, j)] * logp[i, j] for i in range(N) for j in range(K)])
    m += log_likelihood
    for i in range(N):
        m += pulp.lpSum([x[(i, k)] for k in range(K)]) == 1
    for k in range(K):
        m += pulp.lpSum([x[(i, k)] for i in range(N)]) == params.N_CLASSES[k]
    m.solve()
    assert m.status == 1
    x_ast = np.array([[int(x[(i, j)].value()) for j in range(K)] for i in range(N)])
    return x_ast.argmax(axis=1)


class Classifier(nn.Module):
    """
    NLPタスク分類用モデルクラス.

    Parameters:
    -----------
    model_name: str
        使用するモデルの名前.
    typ: str
        NLPモデルから特徴量を取る位置.
    num_classes: int
        学習するデータのクラス数.
    """
    def __init__(self, model_name, typ, num_classes=4):
        super().__init__()

        self.name = model_name
        self.typ = typ
        if model_name in ["albert-large-v2", "xlm-mlm-ende-1024"]:
            nodes = 1024
        else:
            nodes = 768

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        if typ != "ht":
            self.linear = nn.Linear(nodes, num_classes)
        else:
            self.linear = nn.Linear(nodes*2, num_classes)
        nn.init.normal_(self.linear.weight, std=0.02)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if self.name in ["bert-base-cased", "albert-large-v2"]:
            output, _ = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids)
            #output = output[:, 0, :]
        elif self.name in ["xlnet-base-cased", "xlm-mlm-ende-1024"]:
            output = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids)
            output = output[0]
            #output = output[:, 0, :]
        elif self.name in ["roberta-base", "distilbert-base-cased"]:
            output = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask,
                )
            output = output[0]
            #output = output[:, 0, :]
        
        if self.typ == "h":
            output = output[:, 0, :]
        elif self.typ == "m":
            output = torch.mean(output, dim=1)
        elif self.typ == "t" or self.typ=="FRt" or self.typ=="DEt":
            output = output[:, -1, :]
        elif self.typ ==  "ht":
            output = torch.cat((output[:, 0, :], output[:, -1, :]), dim=-1)
        else:
            output = output[:, 0, :]
        output = self.dropout(output)
        output = self.linear(output)
        return output


class Parameters(object):
    """
    パラメータ管理用クラス.
    """
    def __init__(self):
        self.SEED = 2020
        # コードのパス. os.getcwd()が動かない場合はstrで直接渡す.
        #BASE_PATH = "C:/StudentCup2020/2nd/"
        self.BASE_PATH = os.getcwd() + '/'
        self.TEST_FILE = self.BASE_PATH + "data/test.csv"
        self.TEST_FILE_FR = self.BASE_PATH+"data/test_fr.csv"
        self.TEST_FILE_DE = self.BASE_PATH+"data/test_de.csv"
        self.TEXT_COL = "description"
        self.TARGET = "jobflag"
        self.NUM_CLASS = 4
        
        self.LGB_TEST_FILE = self.BASE_PATH+"data/lgb_test.csv"
        self.OUTPUT_PATH = self.BASE_PATH + "outputs/"
        
        if True:
            self.TRAIN_WEIGHT = np.array([0.2129, 0.1187, 0.4695, 0.1989])
            self.TEST_WEIGHT = np.array([0.2314, 0.1833, 0.1982, 0.3872])
        else:
            assert True, "TRAIN, TESTの重みをdf[TARGET].value_counts()/len(df), 予測値から決めてください"
            self.TRAIN_WEIGHT = np.array([0.25, 0.25, 0.25, 0.25])
            self.TEST_WEIGHT = np.array([0.25, 0.25, 0.25, 0.25])

        self.CLASS_WEIGHT = self.TEST_WEIGHT / self.TRAIN_WEIGHT
        self.CLASS_WEIGHT /= sum(self.CLASS_WEIGHT)
        self.CLASS_WEIGHT_TENSOR = torch.tensor(self.CLASS_WEIGHT).cuda()

        len_test = len(pd.read_csv(self.TEST_FILE))
        self.N_CLASSES = np_rounder(len_test*self.TEST_WEIGHT).tolist()
        while sum(self.N_CLASSES) < len_test:
            diff = np.abs(0.5 - len_test*self.TEST_WEIGHT%1)
            self.N_CLASSES[np.argmin(diff)] += 1
        while sum(self.N_CLASSES) > len_test:
            diff = np.abs(0.5 - len_test*self.TEST_WEIGHT%1)
            self.N_CLASSES[np.argmin(diff)] -= 1

        self.DEVICE = "cuda"
        self.MODELS_DIR = self.BASE_PATH + "models/"
        self.NUM_SPLITS = 4
        
        self.VALID_BATCH_SIZE = 128
        self.MAX_TOKEN_LEN = 128
params = Parameters()


def main(params):
    # --- lightgbm --- #
    print("LightGBM Predicting...")
    X_test = pd.read_csv(params.LGB_TEST_FILE)
    y_pred = predict_lgb(X_test, n_folds=params.NUM_SPLITS)
    np.save(params.OUTPUT_PATH+"lgb_yprd", y_pred)
    
    # --- catboost --- #
    print("CatBoost Predicting...")
    test = pd.read_csv(params.TEST_FILE)
    col = [c for c in test.columns if c not in ['id', params.TARGET]]
    X_test = test[col]
    y_pred = predict_ctb(X_test, n_folds=params.NUM_SPLITS)
    np.save(params.OUTPUT_PATH+"cat_yprd", y_pred)
    
    # --- roberta --- #
    model_name = "roberta-base"
    typs = ["h", "m", "t", "ht"]
    for typ in typs:
        print("Robert {} Predicting...".format(typ))
        preds = predict_nlp(model_name, typ, params.TEST_FILE)
        np.save(params.OUTPUT_PATH+model_name+"_"+typ+"_yprd", preds)
    typ = "FRt"
    print("Robert {} Predicting...".format(typ))
    preds = predict_nlp(model_name, typ, params.TEST_FILE_FR)
    np.save(params.OUTPUT_PATH+model_name+"_"+typ+"_yprd", preds)
    typ = "DEt"
    print("Robert {} Predicting...".format(typ))
    preds = predict_nlp(model_name, typ, params.TEST_FILE_DE)
    np.save(params.OUTPUT_PATH+model_name+"_"+typ+"_yprd", preds)

    # --- ensemble --- #
    model_names = ["lgb", "cat",
                   "roberta-base_h", "roberta-base_m", "roberta-base_t", "roberta-base_ht",
                   "roberta-base_FRt", "roberta-base_DEt"]
    test = pd.read_csv(params.TEST_FILE)
    y_pred = np.zeros((test.shape[0], 4, len(model_names)))
    for i, model_name in enumerate(model_names):
        yprd = np.load(params.OUTPUT_PATH+model_name+"_yprd.npy")
        y_pred[:, :, i] = yprd
    best_w = np.load(params.OUTPUT_PATH+"config_ensemble_bestw.npy")
    best_cw = np.load(params.OUTPUT_PATH+"config_ensemble_bestcw.npy")
    test_pred = np.average(y_pred, axis=2, weights=best_w)
    test_pred = test_pred * best_cw

    # --- post processing --- #
    test_pred = sigmoid(test_pred)
    test_pred = test_pred / np.sum(test_pred, axis=1).reshape(test.shape[0], -1)
    test_pred = hack(test_pred) + 1
    
    test = pd.read_csv(params.TEST_FILE)
    submit = pd.DataFrame({'index':test['id'], 'pred':test_pred})
    submit.to_csv(params.BASE_PATH+"data/submission.csv", index=False, header=False)


if __name__ == "__main__":
    seed_everything(params.SEED)
    if "models" not in os.listdir(params.BASE_PATH):
        os.mkdir(params.BASE_PATH + "models/")
    if "outputs" not in os.listdir(params.BASE_PATH):
        os.mkdir(params.BASE_PATH + "outputs/")
    main(params)