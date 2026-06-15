"""5G 用户预测完整建模流程：逻辑回归 / 随机森林 / LightGBM。"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110

SEED = 42
FIG_DIR = 'output/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def cross_validate_auc(model, X, y, n_splits=5):
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    auc_scores = []
    
    # 将数据划分为每折的训练集和验证集进行循环评估
    for train_indices, val_indices in stratified_kfold.split(X, y):
        # 复制模型以确保这一折的训练过程与其它折完全独立
        cloned_model = clone(model)
        
        # 提取当前折的训练数据与验证数据
        X_train_fold = X.iloc[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X.iloc[val_indices]
        y_val_fold = y[val_indices]
        
        # 在训练数据折上拟合模型
        cloned_model.fit(X_train_fold, y_train_fold)
        
        # 获取验证折上预测为正类（5G用户）的概率
        val_predictions = cloned_model.predict_proba(X_val_fold)[:, 1]
        
        # 计算当前折的 AUC 得分
        auc = roc_auc_score(y_val_fold, val_predictions)
        auc_scores.append(auc)
        
    return np.array(auc_scores)


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    
    # 计算 5 折交叉验证的 AUC
    cv_auc_scores = cross_validate_auc(model, X_train, y_train)
    
    # 使用所有训练数据训练最终模型
    final_model = clone(model)
    final_model.fit(X_train, y_train)
    
    # 预测测试集概率
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    
    # 评估测试集性能：AUC 和 AP (Average Precision)
    test_auc = roc_auc_score(y_test, test_probabilities)
    average_precision = average_precision_score(y_test, test_probabilities)
    
    elapsed_time = time.time() - start_time
    
    print(f'[{model_name}] CV AUC = {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f} | '
          f'test AUC = {test_auc:.4f} | AP = {average_precision:.4f} | {elapsed_time:.0f}s')
          
    return {
        'model_name': model_name,
        'cv_scores': cv_auc_scores,
        'test_auc': test_auc,
        'test_predictions': test_probabilities,
        'average_precision': average_precision,
        'final_model': final_model
    }


def perform_exploratory_data_analysis(df, categorical_cols, numerical_cols):
    # 1. 目标变量（类别标签）分布直方图
    target_counts = df['target'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['非5G (0)', '5G (1)'], target_counts.values, color=['#6c8ebf', '#b85450'])
    
    # 在条形图顶部添加数量和百分比标签
    for index, count in enumerate(target_counts.values):
        percentage = count / len(df) * 100
        label_text = f'{count}\n{percentage:.2f}%'
        ax.text(index, count, label_text, ha='center', va='bottom')
        
    ax.set_title('目标变量分布（类别极度不平衡）')
    ax.set_ylabel('样本数')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/01_target_dist.png')
    plt.close()

    # 2. 连续型数值特征与目标变量的 Pearson 相关性分析
    features_to_correlate = numerical_cols + ['target']
    correlation_matrix = df[features_to_correlate].corr()
    target_correlation = correlation_matrix['target'].drop('target')
    
    # 按照相关性绝对值大小进行降序排列
    sorted_correlation = target_correlation.abs().sort_values(ascending=False)
    top_15_features = target_correlation.loc[sorted_correlation.index].head(15)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top_15_features.index[::-1], top_15_features.values[::-1], color='#6c8ebf')
    ax.set_title('数值特征与 target 的相关性 Top15')
    ax.set_xlabel('Pearson 相关系数')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/02_num_corr.png')
    plt.close()

    # 3. 绘制 4 个重要连续特征在不同目标类别下的分布图
    important_features = ['num_37', 'num_3', 'num_10', 'num_30']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    
    for ax, feature_name in zip(axes.flatten(), important_features):
        for class_label, color in [(0, '#6c8ebf'), (1, '#b85450')]:
            class_subset = df[df['target'] == class_label]
            ax.hist(class_subset[feature_name], bins=50, alpha=0.5,
                    color=color, label=f'target={class_label}', density=True)
        ax.set_title(feature_name)
        ax.legend()
        ax.set_yscale('log')
        
    plt.suptitle('重要数值特征在不同类别下的分布')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/03_feat_dist.png')
    plt.close()
    print('EDA 图表已保存')


def compare_models(results, y_test):
    model_names = [res['model_name'] for res in results]
    cv_means = [res['cv_scores'].mean() for res in results]
    cv_stds = [res['cv_scores'].std() for res in results]

    # 绘制左侧 CV 柱状图和右侧 ROC 曲线
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 左图：5折交叉验证的均值与标准差
    axes[0].bar(model_names, cv_means, yerr=cv_stds,
                color=['#8c564b', '#2ca02c', '#1f77b4'], capsize=5)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel('AUC')
    axes[0].set_title('5 折交叉验证 AUC')

    # 右图：测试集上的 ROC 曲线
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res['test_predictions'])
        axes[1].plot(fpr, tpr, label=f"{res['model_name']} ({res['test_auc']:.4f})")
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4)
    axes[1].set_title('测试集 ROC 曲线')
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/04_model_compare.png')
    plt.close()

    # 绘制测试集 Precision-Recall (PR) 曲线
    fig, ax = plt.subplots(figsize=(6, 5))
    for res in results:
        precision, recall, _ = precision_recall_curve(y_test, res['test_predictions'])
        ax.plot(recall, precision, label=f"{res['model_name']} (AP={res['average_precision']:.4f})")
    ax.set_title('测试集 PR 曲线')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/05_pr_curve.png')
    plt.close()

    # 导出模型指标对比汇总表
    summary_df = pd.DataFrame({
        'model': model_names,
        'cv_auc_mean': np.round(cv_means, 4),
        'cv_auc_std': np.round(cv_stds, 4),
        'test_auc': np.round([res['test_auc'] for res in results], 4),
        'test_AP': np.round([res['average_precision'] for res in results], 4),
    })
    summary_df.to_csv('output/cv_results.csv', index=False)
    print('\n===== 模型对比 =====')
    print(summary_df.to_string(index=False))
    return summary_df


def save_final_submission(results, test_ids):
    # 选择测试集 AUC 最高的模型生成预测概率文件
    best_model_result = max(results, key=lambda res: res['test_auc'])
    submission_df = pd.DataFrame({
        'id': test_ids,
        'target': best_model_result['test_predictions']
    })
    submission_df.to_csv('output/submission.csv', index=False)
    print(f"\n最优模型 {best_model_result['model_name']}（test AUC={best_model_result['test_auc']:.4f}），"
          f"预测已写入 output/submission.csv")


def plot_feature_importance(lgb_model):
    # 用 LightGBM 模型的分裂次数评估特征重要性
    feature_importances = pd.Series(
        lgb_model.feature_importances_,
        index=lgb_model.feature_name_
    )
    top_20_importances = feature_importances.sort_values(ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(top_20_importances.index[::-1], top_20_importances.values[::-1], color='#1f77b4')
    ax.set_title('LightGBM 特征重要性 Top20')
    ax.set_xlabel('分裂次数')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/06_lgb_importance.png')
    plt.close()


def main():
    # 加载数据集
    df = pd.read_csv('train.csv')
    print('数据规模:', df.shape)
    
    # 提取特征列
    categorical_cols = [col for col in df.columns if col.startswith('cat_')]
    numerical_cols = [col for col in df.columns if col.startswith('num_')]
    
    # cat_12 包含 11 万种不同取值，不适合做独热编码，将作为连续数值特征处理
    categorical_for_encoding = [col for col in categorical_cols if col != 'cat_12']

    print(f'正样本占比: {df["target"].mean():.4f}')
    print(f'缺失值总数: {df.isnull().sum().sum()}')

    # 执行 EDA 分析
    perform_exploratory_data_analysis(df, categorical_cols, numerical_cols)

    # 提取标签与特征
    y = df['target'].astype(int).values
    X = df.drop(columns=['id', 'target'])
    sample_ids = df['id'].values

    # 将数据集划分为 80% 训练集与 20% 测试集
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, sample_ids, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f'train {len(X_train)}, test {len(X_test)}（test 正样本 {int(y_test.sum())}）')

    evaluation_results = []

    # 1. 逻辑回归模型（线性基线）
    # 数据转换预处理：对类别特征进行独热编码，数值型和 cat_12 进行标准化缩放
    lr_preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_for_encoding),
        ('scaler', StandardScaler(), numerical_cols + ['cat_12']),
    ])
    lr_pipeline = Pipeline([
        ('preprocessor', lr_preprocessor),
        ('classifier', LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', n_jobs=-1))
    ])
    
    lr_results = evaluate_model(
        'LogisticRegression', lr_pipeline, X_train, y_train, X_test, y_test
    )
    evaluation_results.append(lr_results)

    # 2. 随机森林模型（Bagging）
    # 数据转换预处理：对类别特征进行序数编码，数值型不做处理
    rf_preprocessor = ColumnTransformer([
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_for_encoding),
        ('numeric', 'passthrough', numerical_cols + ['cat_12']),
    ])
    rf_pipeline = Pipeline([
        ('preprocessor', rf_preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=120, max_depth=25, min_samples_leaf=10,
            class_weight='balanced', n_jobs=-1, random_state=SEED
        ))
    ])
    
    rf_results = evaluate_model(
        'RandomForest', rf_pipeline, X_train, y_train, X_test, y_test
    )
    evaluation_results.append(rf_results)

    # 3. LightGBM 模型（Boosting，原生支持类别特征）
    # 特征类型转换：显式地将类别特征转换成 category 数据类型
    X_train_lgb = X_train.copy()
    X_test_lgb = X_test.copy()
    for col in categorical_for_encoding:
        X_train_lgb[col] = X_train_lgb[col].astype('category')
        X_test_lgb[col] = X_test_lgb[col].astype('category')
        
    # 计算正负样本权重比例
    positive_weight = (y_train == 0).sum() / (y_train == 1).sum()
    lgbm_classifier = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=positive_weight, random_state=SEED,
        n_jobs=-1, verbose=-1
    )
    
    lgbm_results = evaluate_model(
        'LightGBM', lgbm_classifier, X_train_lgb, y_train, X_test_lgb, y_test
    )
    evaluation_results.append(lgbm_results)

    # 汇总对比模型并在测试集上生成图表
    compare_models(evaluation_results, y_test)
    
    # 保存最优模型的预测结果到 submission.csv
    save_final_submission(evaluation_results, ids_test)
    
    # 绘制 LightGBM 的特征重要性分布
    best_model = evaluation_results[-1]['final_model']
    plot_feature_importance(best_model)
    
    print('\n全部完成')


if __name__ == '__main__':
    main()
