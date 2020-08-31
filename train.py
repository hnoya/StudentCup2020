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


def metric_f1(labels, preds):
    """
    クラスごとに重みづけしたF1評価関数.

    Parameters:
    -----------
    labels: np.array
        正解ラベル.
    preds: np.array
        予測ラベル. 予測確率ではないことに注意.

    Returns:
    -----------
    score: params.CLASS_WEIGHTで重みづけされたF1スコア.
    """
    return f1_score(labels, preds, average=None) @ params.CLASS_WEIGHT


def metric_f1_lgb(preds, data):
    """
    lightgbmのためのF1評価関数.
    詳細はlightgbmのドキュメント参照.
    
    Parameters:
    -----------
    preds: np.array
        予測値, flattenされているのでreshapeする必要あり.
    data: lightgbm.Dataset
        学習データ.
    
    Returns:
    -----------
    "metric_f1": str
        評価関数名.
    score: float
        スコア.
    True: bool
        評価値が高い方が良いモデルか否か.
        損失関数を使う場合はFalse.
    """
    y_true = data.get_label()
    preds = preds.reshape(params.NUM_CLASS, len(preds) // params.NUM_CLASS)
    y_pred = np.argmax(preds, axis=0)
    score = f1_score(y_true, y_pred, average=None) @ params.CLASS_WEIGHT
    return "metric_f1", score, True


def make_weight(x):
    """
    Lightgbmのための重みづけ関数.
    
    Parameters:
    -----------
    x: int
        ラベル番号.
    
    Returns:
    -----------
    params.CLASS_WEIGHT[x]: float
        対応するラベルの重み.
    """
    return params.CLASS_WEIGHT[x]


def make_folded_df(csv_file, num_splits=4):
    """
    fold番号を振るための関数.
    StratifiedKFoldを使用するため、labelsという列名でラベルを保持する必要がある.

    Parameters:
    -----------
    csv_file: str
        csvファイルのパス.
    num_splits: int
        フォールド数.
    
    Returns:
    -----------
    df: pd.DataFrame
        foldにフォールド番号が入ったdf.
    """
    df = pd.read_csv(csv_file)
    df[params.TARGET] = df[params.TARGET] - 1
    df["fold"] = -1
    df = df.rename(columns={params.TARGET: 'labels'})
    label = df["labels"].tolist()

    skfold = StratifiedKFold(num_splits, shuffle=True, random_state=params.SEED)
    for fold, (train_index, valid_index) in enumerate(skfold.split(range(len(label)), label)):
        df['fold'].iloc[valid_index] = fold
    return df


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


def train_lgb(X, y, weight, n_folds=4):
    """
    lightgbm用の訓練関数.
    
    Parameters:
    -----------
    X: pd.DataFrame
        訓練用の説明変数.
    y: pd.DataFrame
        訓練用の被説明変数.
    weight: List[float]
        訓練時のサンプルの重み.
    n_folds: int
        フォールド数.
    
    Returns:
    -----------
    scores: float
        訓練時のOOFスコア.
    feature_importances: pd.DataFrame
        モデルの特徴量の重要度.
    train_pred: np.array
        訓練時のOOF予測値.
    """
    train_pred = np.zeros((X.shape[0], y.nunique()), dtype='float32')
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X.columns
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=params.SEED)

    print("LightGBM Training...")
    for fold, (train_idx, valid_idx) in enumerate(tqdm(kfold.split(X, y))):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        weight_train, weight_valid = weight.iloc[train_idx], weight.iloc[valid_idx]
        train_data = lgb.Dataset(X_train, label=y_train, weight=weight_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, weight=weight_valid)
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'None',
            'learning_rate': 0.01,
            'max_depth': -1,
            'num_leaves': 31,
            'max_bin': 31,
            'min_data_in_leaf': 3,
            'verbose': -1,
            'seed': params.SEED,
            'drop_seed': params.SEED,
            'data_random_seed':params.SEED
        }
        model = lgb.train(lgb_params, train_data, valid_sets=[train_data,valid_data],
                          num_boost_round=params.GBDT_ROUNDS,
                          early_stopping_rounds=params.GBDT_EARLY_STOPPING,
                          feval=metric_f1_lgb,
                          verbose_eval=False, )
        pickle.dump(model, open(params.MODELS_DIR+"lgb_fold{}.lgbmodel".format(fold),
                                "wb"))
        y_val_pred = model.predict(X_valid)
        train_pred[valid_idx,:] = y_val_pred
        feature_importances['fold_{}'.format(fold)] = model.feature_importance(importance_type='gain')
        gc.collect()

    feature_importances['importance'] = feature_importances.iloc[:,1:1+n_folds].mean(axis=1)
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    scores = f1_score(y, np.argmax(train_pred, axis=1), average=None) @ params.CLASS_WEIGHT
    return scores, feature_importances, train_pred


def train_ctb(X, y, n_folds=4):
    """
    catboost用の訓練関数.
    
    Parameters:
    -----------
    X: pd.DataFrame
        訓練用の説明変数.
    y: pd.DataFrame
        訓練用の被説明変数.
    n_folds: int
        フォールド数.
    
    Returns:
    -----------
    scores: float
        訓練時のOOFスコア.
    train_pred: np.array
        訓練時のOOF予測値.
    """
    train_pred = np.zeros((X.shape[0], y.nunique()), dtype='float32')
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=params.SEED)

    print("CatBoost Training...")
    for fold, (train_idx, valid_idx) in enumerate(tqdm(kfold.split(X, y))):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        train_data = Pool(X_train, label=y_train, text_features=[params.TEXT_COL])
        valid_data = Pool(X_valid, label=y_valid, text_features=[params.TEXT_COL])
        ctb_params = {
            'objective': 'MultiClass',
            'loss_function': 'TotalF1',
            'class_weights': params.CLASS_WEIGHT.tolist(),
            'num_boost_round':params.GBDT_ROUNDS,
            'early_stopping_rounds':params.GBDT_EARLY_STOPPING,
            'learning_rate':0.03,
            'l2_leaf_reg':3.0,
            #'subsample':0.66,
            'max_depth':6,
            'grow_policy':'SymmetricTree',
            'min_data_in_leaf':1,
            'max_leaves':31,
            'verbose':False,
            'random_seed':params.SEED,
        }
        model = CatBoost(ctb_params)
        model.fit(train_data, eval_set=[valid_data], use_best_model=True, plot=False)
        pickle.dump(model, open(params.MODELS_DIR+"ctb_fold{}.ctbmodel".format(fold),
                                "wb"))
        train_pred[valid_idx, :] = model.predict(X_valid)
        gc.collect()
    scores = f1_score(y, np.argmax(train_pred, axis=1), average=None) @ params.CLASS_WEIGHT
    return scores, train_pred


def train_fn(dataloader, model, criterion, optimizer, device, epoch):
    """
    NLPモデル訓練EPOCH用関数.

    Parameters:
    -----------
    dataloader: torch.dataset.dataloader
        NLP用のデータローダー.
    model: torch.nn.Module
        NLP用のtorchのモデル.
    criterion: torch.nn.*Loss
        NLP用の損失関数. 自分で作成した関数も可能.
    optimizer: torch.optim.*
        NLP用の最適化関数.
    device: str
        使用するデバイス. "cuda" or "cpu".
    epoch: int
        学習するエポック数.
    
    Returns:
    ---------
    train_losses: float
        訓練時の累積損失.
    train_acc: float
        訓練時の正解率.
    train_f1: float
        訓練時のF1.
    """
    model.train()
    train_losses = 0
    correct_counts = 0
    train_labels = []
    train_preds = []
    for i, batch in enumerate(dataloader):
        if len(batch.values())==4:
            attention_mask, input_ids, labels, token_type_ids = batch.values()
        else:
            attention_mask, input_ids, labels = batch.values()
            token_type_ids = None
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, axis=1)
        loss.backward()
        optimizer.step()

        train_losses += loss.item()
        correct_counts += torch.sum(preds == labels)

        train_labels += labels.tolist()
        train_preds += preds.tolist()

    train_losses = train_losses / len(dataloader)
    train_acc = correct_counts.double().cpu().detach().numpy() / len(dataloader.dataset)
    train_f1 = metric_f1(train_labels, train_preds)

    return train_losses, train_acc, train_f1


def eval_fn(dataloader, model, criterion, device):
    """
    NLPモデル検証EPOCH用関数.

    Parameters:
    -----------
    dataloader: torch.dataset.dataloader
        NLP用のデータローダー.
    model: torch.nn.Module
        NLP用のtorchのモデル.
    criterion: torch.nn.*Loss
        NLP用の損失関数. 自分で作成した関数も可能.
    device: str
        使用するデバイス. "cuda" or "cpu".

    Returns:
    ---------
    valid_losses: float
        検証時の累積損失.
    valid_acc: float
        検証時の正解率.
    valid_f1: float
        検証時のF1.
    """
    model.eval()
    valid_losses = 0
    total_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if len(batch.values())==4:
                attention_mask, input_ids, labels, token_type_ids = batch.values()
            else:
                attention_mask, input_ids, labels = batch.values()
                token_type_ids = None
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            valid_losses += loss.item()
            total_corrects += torch.sum(preds == labels)
            all_labels += labels.tolist()
            all_preds += preds.tolist()

    valid_losses = valid_losses / len(dataloader)
    valid_acc = total_corrects.double().cpu().detach().numpy() / len(dataloader.dataset)

    valid_f1 = metric_f1(all_labels, all_preds)

    return valid_losses, valid_acc, valid_f1


def trainer(fold, df, model_name, oof_pred, typ):
    """
    NLP訓練全Fold用関数.
    F1で保存する.

    Parameters:
    -----------
    fold: int
        検証に使用するフォールドの番号.
    df: pd.DataFrame
        学習に使用するデータフレーム.
    model_name: str
        NLPモデルの名前.
    oof_pred: np.array
        OOF予測値.
    typ: str
        NLPモデルから特徴量を取る位置.

    Returns:
    ----------
    best_f1: float
        保存したモデルのF1.
    oof_pred: np.array
        OOF予測値.
    """
    train_df = df[df.fold != fold].reset_index(drop=True)
    valid_df = df[df.fold == fold].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = make_dataset(train_df, tokenizer, params.DEVICE, model_name)
    valid_dataset = make_dataset(valid_df, tokenizer, params.DEVICE, model_name)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.TRAIN_BATCH_SIZE, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=params.VALID_BATCH_SIZE, shuffle=False
    )

    model = Classifier(model_name, typ, num_classes=params.NUM_CLASS)
    model = model.to(params.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=params.CLASS_WEIGHT_TENSOR.float())
    optimizer = AdamW(model.parameters(), lr=2e-5)

    train_losses = []
    train_accs = []
    train_f1s = []
    valid_losses = []
    valid_accs = []
    valid_f1s = []

    best_loss = np.inf
    best_acc = 0
    best_f1 = 0

    for epoch in range(params.EPOCHS):
        train_loss, train_acc, train_f1 = train_fn(train_dataloader, model, criterion, optimizer, params.DEVICE, epoch)
        valid_loss, valid_acc, valid_f1 = eval_fn(valid_dataloader, model, criterion, params.DEVICE)
        #print(f"Loss: {valid_loss}  Acc: {valid_acc}  f1: {valid_f1}  ", end="")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_f1s.append(valid_f1)

        best_loss = valid_loss if valid_loss < best_loss else best_loss
        besl_acc = valid_acc if valid_acc > best_acc else best_acc
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            #print("model saving!", end="")
            torch.save(model.state_dict(), params.MODELS_DIR + f"best_{model_name}_{typ}_{fold}.pth")
        #print("\n")

    valid_pred = []
    model.load_state_dict(torch.load(params.MODELS_DIR + f"best_{model_name}_{typ}_{fold}.pth"))
    model.to(params.DEVICE)
    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            if len(batch.values())==4:
                attention_mask, input_ids, labels, token_type_ids = batch.values()
            else:
                attention_mask, input_ids, labels = batch.values()
                token_type_ids = None
            outputs = model(input_ids, attention_mask, token_type_ids)
            valid_pred += outputs.tolist()
    oof_pred[df[df.fold == fold].index, :] = valid_pred
    return best_f1, oof_pred


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
    パラメータ管理用のクラス.
    """
    def __init__(self):
        self.SEED = 2020
        # コードのパス. os.getcwd()が動かない場合はstrで直接渡す.
        #BASE_PATH = "C:/StudentCup2020/2nd/"
        self.BASE_PATH = os.getcwd() + '/'
        self.TRAIN_FILE = self.BASE_PATH + "data/train.csv"
        self.TRAIN_FILE_FR = self.BASE_PATH+"data/train_fr.csv"
        self.TRAIN_FILE_DE = self.BASE_PATH+"data/train_de.csv"
        self.TEXT_COL = "description"
        self.TARGET = "jobflag"
        self.NUM_CLASS = 4
        
        self.LGB_TRAIN_FILE = self.BASE_PATH+"data/lgb_train.csv"
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
        
        self.DEVICE = "cuda"
        self.MODELS_DIR = self.BASE_PATH + "models/"
        self.EPOCHS = 5
        self.GBDT_ROUNDS = 2000
        self.GBDT_EARLY_STOPPING = 100
        self.NUM_SPLITS = 4
        
        self.TRAIN_BATCH_SIZE = 32
        self.VALID_BATCH_SIZE = 128
        self.MAX_TOKEN_LEN = 128
params = Parameters()


def main(params):
    # --- lightgbm --- #
    lgb_df = pd.read_csv(params.LGB_TRAIN_FILE)
    X = lgb_df.drop([params.TARGET], axis=1)
    y = lgb_df[params.TARGET] - 1
    weight = y.apply(lambda x: make_weight(x))
    scores, feature_importances, train_pred = train_lgb(X, y, weight, n_folds=params.NUM_SPLITS)
    print("LightGBM Score: {}".format(scores))
    feature_importances.to_csv(params.OUTPUT_PATH+"lgb_feature_importances.csv", index=False)
    np.save(params.OUTPUT_PATH+"lgb_trap", train_pred)

    # --- catboost --- #
    train = pd.read_csv(params.TRAIN_FILE).drop(['id'], axis=1)
    train[params.TARGET] -= 1
    col = [c for c in train.columns if c not in ['id', params.TARGET]]
    X = train[col]
    y = train[params.TARGET].astype(int)
    scores, train_pred = train_ctb(X, y, n_folds=params.NUM_SPLITS)
    print("CatBoost Score: {}".format(scores))
    np.save(params.OUTPUT_PATH+"cat_trap", train_pred)

    # --- roberta --- #
    print("roberta Training...")
    df = make_folded_df(params.TRAIN_FILE, params.NUM_SPLITS)
    model_name = "roberta-base"
    typs = ["h", "m", "t", "ht"]
    for typ in typs:
        f1_scores = []
        oof_pred = np.zeros((len(df), params.NUM_CLASS), dtype='float32')
        print("="*10 + "roberta {} Training".format(typ) + "="*10)
        for fold in tqdm(range(params.NUM_SPLITS)):
            f1, oof_pred = trainer(fold, df, model_name, oof_pred, typ)
            f1_scores.append(f1)
        scores = metric_f1(df['labels'], np.argmax(oof_pred, axis=1))
        print("roberta {} Score: {}".format(typ, scores))
        np.save(params.OUTPUT_PATH+model_name+"_"+typ+"_trap", oof_pred)

    df = make_folded_df(params.TRAIN_FILE_FR, params.NUM_SPLITS)
    model_name = "roberta-base"
    typ = "FRt"
    f1_scores = []
    oof_pred = np.zeros((len(df), params.NUM_CLASS), dtype='float32')
    print("="*10 + "roberta {} Training".format(typ) + "="*10)
    for fold in range(params.NUM_SPLITS):
        f1, oof_pred = trainer(fold, df, model_name, oof_pred, typ)
        f1_scores.append(f1)
    scores = metric_f1(df['labels'], np.argmax(oof_pred, axis=1))
    print("roberta {} Score: {}".format(typ, scores))
    np.save(params.OUTPUT_PATH+model_name+"_"+typ+"_trap", oof_pred)

    df = make_folded_df(params.TRAIN_FILE_DE, params.NUM_SPLITS)
    model_name = "roberta-base"
    typ = "DEt"
    f1_scores = []
    oof_pred = np.zeros((len(df), params.NUM_CLASS), dtype='float32')
    print("="*10 + "roberta {} Training".format(typ) + "="*10)
    for fold in range(params.NUM_SPLITS):
        f1, oof_pred = trainer(fold, df, model_name, oof_pred, typ)
        f1_scores.append(f1)
    scores = metric_f1(df['labels'], np.argmax(oof_pred, axis=1))
    print("roberta {} Score: {}".format(typ, scores))
    np.save(params.OUTPUT_PATH+model_name+"_"+typ+"_trap", oof_pred)

    # --- ensemble --- #
    print("Ensemble...")
    model_names = ["lgb", "cat",
                   "roberta-base_h", "roberta-base_m", "roberta-base_t", "roberta-base_ht",
                   "roberta-base_FRt", "roberta-base_DEt"]
    train = pd.read_csv(params.TRAIN_FILE)
    train["label"] = train[params.TARGET] - 1
    train_pred = np.zeros((train.shape[0], 4, len(model_names)))
    for i, model_name in enumerate(model_names):
        trap = np.load(params.OUTPUT_PATH+model_name+"_trap.npy")
        train_pred[:, :, i] = trap

    best_w = np.ones(len(model_names))
    best_w /= sum(best_w)
    trap = np.average(train_pred, axis=2, weights=best_w)
    best_cw = 0.5 + np.ones(4)
    best_cw /= sum(best_cw)
    trap *= best_cw
    best_score = f1_score(train['label'], np.argmax(trap, axis=1), average=None) @ params.CLASS_WEIGHT
    for i in tqdm(range(100_000)):
        w = np.random.random(len(model_names))
        w /= sum(w)
        trap = np.average(train_pred, axis=2, weights=w)
        cw = 0.5 + np.random.random(4)
        cw /= sum(cw)
        trap = trap * cw
        score = f1_score(train['label'], np.argmax(trap, axis=1), average=None) @ params.CLASS_WEIGHT
        if score > best_score:
            best_score = score
            best_w = w
            best_cw = cw
    print("Best Ensemble Score: {}".format(best_score))
    oof_pred = np.average(train_pred, axis=2, weights=best_w)
    oof_pred = oof_pred * best_cw
    np.save(params.OUTPUT_PATH+"trap_ensemble", oof_pred)
    np.save(params.OUTPUT_PATH+"config_ensemble_bestw", best_w)
    np.save(params.OUTPUT_PATH+"config_ensemble_bestcw", best_cw)


if __name__ == "__main__":
    seed_everything(params.SEED)
    if "models" not in os.listdir(params.BASE_PATH):
        os.mkdir(params.BASE_PATH + "models/")
    if "outputs" not in os.listdir(params.BASE_PATH):
        os.mkdir(params.BASE_PATH + "outputs/")
    main(params)