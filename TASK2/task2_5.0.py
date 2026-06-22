# =========================
# 1. 导入依赖
# =========================
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 将 display 替换为 print，保证在 .py 文件中可运行
# 但保留 display 导入以便兼容，若运行环境不支持则回退
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

# 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# 数据加载
csv_path = Path("heart_failure_clinical_records_dataset.csv")
if not csv_path.exists():
    alt_path = Path("45a2ba16-e99f-4a30-9720-f19fc03098ce.csv")
    if alt_path.exists():
        csv_path = alt_path
    else:
        raise FileNotFoundError("未找到数据文件，请将 CSV 放在 Notebook 同目录下。")

print(f"当前使用的数据文件：{csv_path.resolve()}")
df = pd.read_csv(csv_path)
print("数据读取成功。")

# =========================
# 2. 基本查看与描述性统计（保留 time）
# =========================
print("数据维度：", df.shape)
print(df.head())

print("\n数据类型：")
print(df.dtypes.to_frame("dtype"))

print("\n描述性统计：")
print(df.describe(include="all").T)

print("\n目标变量分布：")
print(df["DEATH_EVENT"].value_counts().rename_axis("DEATH_EVENT").to_frame("count"))

print("\n缺失值检查：")
missing = df.isna().sum()
print(missing.to_frame("missing_count"))
if missing.sum() == 0:
    print("结论：该数据集没有缺失值，无需插补。")
else:
    print("结论：存在缺失值，后续需要处理。")

# =========================
# 3. 异常值检测（仅报告，不修改数据）—— 缺陷4修改
# =========================
continuous_cols = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]

def iqr_bounds(series: pd.Series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper

# 不再对数据做任何截断，仅保留原始 df
df_clean = df.copy()  # 实际未修改

outlier_report = []
for col in continuous_cols:
    lower, upper = iqr_bounds(df_clean[col])
    outlier_count = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    outlier_report.append({
        "feature": col,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": int(outlier_count),
    })
outlier_df = pd.DataFrame(outlier_report)
print("\n异常值检测报告（数据未做任何截断）：")
print(outlier_df)

# 注意：以下不再对 df_clean[col] 进行 clip

# =========================
# 4. 相关性热力图（保留 time）
# =========================
corr = df_clean.corr(numeric_only=True)
plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, square=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap (Including Time)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("相关性热力图已保存为 correlation_heatmap.png")

# =========================
# 5. 划分训练集 / 测试集（删除 time 用于建模）
# =========================
X = df_clean.drop(columns=["DEATH_EVENT", "time"])
y = df_clean["DEATH_EVENT"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)
print("训练集目标分布：")
print(y_train.value_counts().rename_axis("DEATH_EVENT").to_frame("count"))
print("测试集目标分布：")
print(y_test.value_counts().rename_axis("DEATH_EVENT").to_frame("count"))
print("建模使用的特征（不含 time）：", list(X.columns))

# =========================
# 6. 定义 MLP 模型（PyTorch，兼容 sklearn），增加 Early Stopping —— 缺陷3修改
# =========================
class PyTorchMLP(BaseEstimator, ClassifierMixin):
    """多层感知机，使用 PyTorch 实现，支持 sklearn 接口，增加验证集和早停"""
    def __init__(self, hidden_sizes=[64, 32], activation='relu',
                 learning_rate=0.001, epochs=100, batch_size=16,
                 random_state=42, verbose=True, patience=15):
        ##### ========== MLP 超参数说明（调参指南） ==========
        ##### hidden_sizes : list，每层神经元个数，例如 [64,32] 表示两层隐藏层。
        #####    - 影响：层数越多/神经元越多，模型容量越大，易过拟合（尤其小样本）。
        #####    - 调参建议：对于299样本，建议保持1~2层，每层不超过64个神经元。可尝试 [32] 或 [64]。
        ##### activation : str，激活函数，'relu' 或 'tanh'。
        #####    - 影响：ReLU 收敛快，但可能造成神经元死亡；Tanh 输出中心对称，有时更稳定。
        #####    - 调参建议：通常 ReLU 是首选，若效果不佳可尝试 tanh。
        ##### learning_rate : float，学习率，控制参数更新步长。
        #####    - 影响：过大导致震荡不收敛，过小收敛极慢或陷入局部最优。
        #####    - 调参建议：常用 1e-3 ~ 1e-2，本数据小，建议 1e-3 或 5e-4。
        ##### epochs : int，最大训练轮数（早停会提前终止）。
        #####    - 影响：太多容易过拟合，太少欠拟合。
        #####    - 调参建议：设大一些（如200），依靠早停停止，无需精细调整。
        ##### batch_size : int，批量大小。
        #####    - 影响：小批量引入噪声有助于泛化，但过小训练慢；大批量训练稳定但可能陷入局部极小。
        #####    - 调参建议：本数据小，推荐 8~32，常用 16。
        ##### patience : int，早停耐心值，验证集损失连续多少轮不下降则停止。
        #####    - 影响：太小容易提前停止（欠拟合），太大则可能过拟合。
        #####    - 调参建议：小样本建议 10~20，常用 15。
        ##### ================================================
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(random_state)

    def _build_model(self, input_dim):
        layers = []
        prev_dim = input_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def fit(self, X, y):
        # 将训练集再拆分为训练子集和验证集 (8:2, 分层)
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=self.random_state
        )

        # 转为 Tensor
        X_train_tensor = torch.tensor(X_train_sub.values if isinstance(X_train_sub, pd.DataFrame) else X_train_sub,
                                      dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_sub.values if isinstance(y_train_sub, pd.Series) else y_train_sub,
                                      dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val,
                                    dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.values if isinstance(y_val, pd.Series) else y_val,
                                    dtype=torch.float32).view(-1, 1).to(self.device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X_train_tensor.shape[1]
        self.model_ = self._build_model(input_dim).to(self.device)
        self.criterion_ = nn.BCELoss()
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        best_loss = np.inf
        best_weights = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer_.zero_grad()
                outputs = self.model_(batch_X)
                loss = self.criterion_(outputs, batch_y)
                loss.backward()
                self.optimizer_.step()
                total_loss += loss.item() * batch_X.size(0)
            avg_train_loss = total_loss / len(dataset)

            # 验证集评估
            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(X_val_tensor)
                val_loss = self.criterion_(val_outputs, y_val_tensor).item()

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early Stopping 检查
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(self.model_.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        # 恢复最佳权重
        if best_weights is not None:
            self.model_.load_state_dict(best_weights)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        X_tensor = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X,
                                dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.model_(X_tensor).cpu().numpy().flatten()
        return np.column_stack((1 - probs, probs))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# =========================
# 7. 定义所有模型（包含 MLP）—— 缺陷2修改：加权重
# =========================
# 计算类别权重供 XGBoost 使用
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive

# ----- 用于交叉验证的模型（不包括 MLP，因为 CV 耗时较长） -----
cv_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.5,
            solver='liblinear',
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced"  # 自动平衡类别权重
        ))
    ]),
    ##### ========== Logistic Regression 超参数说明 ==========
    ##### C : 正则化强度的倒数（默认为1.0），越小正则化越强（防止过拟合）。
    #####    - 调参建议：小样本建议增大正则化（C=0.1~0.5），可尝试 C=0.1, 0.5, 1.0。
    ##### penalty : 正则化类型，'l2'（默认）或 'l1'（产生稀疏解）。
    #####    - 调参建议：一般用 l2，若特征较多可尝试 l1 做特征选择。
    ##### solver : 优化算法，'liblinear'（小数据适用）, 'lbfgs'（多类）等。
    #####    - 调参建议：小样本建议 'liblinear' 或 'lbfgs'。
    ##### max_iter : 最大迭代次数，确保收敛。
    ##### class_weight : 类别权重，'balanced' 或自定义字典，处理不平衡。
    ##### ===================================================

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),
    ##### ========== Random Forest 超参数说明 ==========
    ##### n_estimators : 树的数量，越多越稳定但训练变慢。
    #####    - 调参建议：小样本 100~500 足够，300 是常用值。
    ##### max_depth : 树的最大深度，默认不限制（易过拟合）。
    #####    - 调参建议：小样本强烈建议限制深度，如 max_depth=5~10，防止每棵树过拟合。
    ##### min_samples_split : 内部节点再划分所需最小样本数，默认2。
    #####    - 调参建议：增大可防止过拟合，推荐 5~20。
    ##### min_samples_leaf : 叶子节点最少样本数，默认1。
    #####    - 调参建议：增大可平滑模型，推荐 2~5。
    ##### max_features : 每次分裂考虑的特征数，默认 'sqrt'（即 sqrt(n_features)）。
    #####    - 调参建议：可尝试 'log2' 或 0.3~0.5 比例，增加随机性。
    ##### class_weight : 处理不平衡，'balanced' 或 'balanced_subsample'。
    ##### ===============================================

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    )
    ##### ========== XGBoost 超参数说明 ==========
    ##### n_estimators : 树的棵数（弱学习器数量）。
    #####    - 调参建议：小样本 100~500，结合 early_stopping_rounds 自动停止。
    ##### learning_rate (eta) : 学习率/收缩步长，控制每棵树的贡献。
    #####    - 影响：越小越需要更多树，但泛化更好。常用 0.01~0.3。
    #####    - 调参建议：本数据用 0.05 合理，可尝试 0.01~0.1。
    ##### max_depth : 树的最大深度，控制模型复杂度。
    #####    - 调参建议：小样本强烈建议设为 2~5，防止过拟合。3 是安全值。
    ##### subsample : 每棵树使用的训练样本比例（行采样）。
    #####    - 调参建议：0.7~1.0，可防止过拟合，本数据 0.9 合适。
    ##### colsample_bytree : 每棵树使用的特征比例（列采样）。
    #####    - 调参建议：0.7~1.0，增加随机性，本数据 0.9 合适。
    ##### scale_pos_weight : 正负样本权重比，处理不平衡。
    #####    - 调参建议：通常设为 sum(neg)/sum(pos)，本数据约为2.1。
    ##### reg_lambda (L2正则化) : 权重的L2正则化系数，默认1。
    #####    - 调参建议：增大可防止过拟合，小样本可尝试 1~5。
    ##### reg_alpha (L1正则化) : 权重的L1正则化系数，默认0。
    #####    - 调参建议：若特征较多可尝试非零值做特征选择。
    ##### gamma : 分裂所需最小损失下降，默认0，增大可防止过拟合。
    #####    - 调参建议：小样本可尝试 0.1~0.5。
    ##### ===========================================
}

# ----- 全部模型（含 MLP，用于最终测试集评估） -----
all_models = {
     "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.5,                     # 对齐 cv_models
            solver='liblinear',
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,                 # 对齐 cv_models
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    ),
    "MLP (PyTorch)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", PyTorchMLP(
            hidden_sizes=[64, 32],
            epochs=100,
            batch_size=16,
            verbose=True,
            random_state=RANDOM_STATE,
            patience=15
        ))
    ])
    ##### MLP 的超参数已经在 PyTorchMLP 类的 __init__ 中注释，此处不再重复。
}

# =========================
# 8. 定义交叉验证评估函数 —— 缺陷1
# =========================
def evaluate_cv(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring={"auc": "roc_auc", "recall": "recall"},
        n_jobs=-1
    )
    return {
        "AUC_mean": scores["test_auc"].mean(),
        "AUC_std": scores["test_auc"].std(),
        "Recall_mean": scores["test_recall"].mean(),
        "Recall_std": scores["test_recall"].std()
    }

# =========================
# 9. 进行交叉验证并选择最佳模型
# =========================
print("\n========== 5-Fold CV ==========")
cv_results = []

for name, model in cv_models.items():
    print(f"评估模型：{name} (CV)")
    cv_score = evaluate_cv(model, X_train, y_train)
    cv_results.append({
        "Model": name,
        "AUC_mean": cv_score["AUC_mean"],
        "AUC_std": cv_score["AUC_std"],
        "Recall_mean": cv_score["Recall_mean"],
        "Recall_std": cv_score["Recall_std"]
    })

cv_results_df = pd.DataFrame(cv_results)
print("\n交叉验证结果（按 AUC 均值排序）：")
print(cv_results_df.sort_values("AUC_mean", ascending=False).to_string(index=False))

# 根据 CV AUC 选择最佳模型
best_model_name = cv_results_df.sort_values("AUC_mean", ascending=False).iloc[0]["Model"]
print(f"\n根据交叉验证 AUC 选择的最佳模型：{best_model_name}")

# =========================
# 10. 训练所有模型（用于后续测试集评估和ROC）
# =========================
results = []
roc_data = {}
fitted_models = {}

for name, model in all_models.items():
    print(f"\n训练模型：{name}")
    model.fit(X_train, y_train)
    fitted_models[name] = model

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "AUC": auc
    })
    roc_data[name] = (fpr, tpr, auc)

results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False).reset_index(drop=True)
print("\n=== 测试集评估结果 ===")
print(results_df.to_string(index=False))

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC 曲线对比")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("ROC 曲线已保存为 roc_curves.png")

# =========================
# 11. XGBoost 特征重要性条形图（标注医学含义）
# =========================
feature_meaning = {
    "age": "年龄",
    "anaemia": "贫血",
    "creatinine_phosphokinase": "肌酸激酶",
    "diabetes": "糖尿病",
    "ejection_fraction": "射血分数",
    "high_blood_pressure": "高血压",
    "platelets": "血小板",
    "serum_creatinine": "血清肌酐",
    "serum_sodium": "血清钠",
    "sex": "性别",
    "smoking": "吸烟"
}

xgb_model = fitted_models["XGBoost"]
importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importance_df = importances.reset_index()
importance_df.columns = ["feature", "importance"]
importance_df["medical_meaning"] = importance_df["feature"].map(feature_meaning)

print("\nXGBoost 特征重要性排名（含医学含义）：")
print(importance_df)

top3 = importance_df.head(3)
print("\n对 DEATH_EVENT 影响最大的前 3 个特征（基于XGboost）：")
print(top3)

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
y_labels = [f"{row['feature']} ({row['medical_meaning']})" for _, row in importance_df.iterrows()]
ax.set_yticklabels(y_labels)
plt.xlabel("重要性分数")
plt.title("XGBoost 特征重要性（含医学含义）")
plt.tight_layout()
plt.savefig("xgb_feature_importance_with_meaning.png", dpi=300, bbox_inches="tight")
plt.close()
print("特征重要性图（含医学含义）已保存为 xgb_feature_importance_with_meaning.png")


# =========================
# 11. 最佳模型（Logistic Regression）特征系数与临床影响
# =========================
# 提取最终最佳模型（基于 CV 选出的 LR）的系数
best_lr_pipeline = fitted_models["Logistic Regression"]
# 从 Pipeline 中取出标准化器和模型
scaler = best_lr_pipeline.named_steps["scaler"]
lr_model = best_lr_pipeline.named_steps["model"]

# 获取标准化后的系数（因为数据经过了 StandardScaler，系数可直接比较大小）
coefficients = lr_model.coef_[0]
feature_names = X.columns

# 构建系数 DataFrame
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "medical_meaning": [feature_meaning.get(f, f) for f in feature_names]
})

# 计算 Odds Ratio (OR) = exp(coef)，并排序
coef_df["OR"] = np.exp(coef_df["coefficient"])
# 系数绝对值越大，对结局影响越大
coef_df["abs_coef"] = np.abs(coef_df["coefficient"])

# 按系数绝对值排序（即特征重要性排序）
coef_df_sorted = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

print("\n=== 最佳模型（Logistic Regression）特征影响排名（按标准化系数绝对值） ===")
print(coef_df_sorted[["feature", "medical_meaning", "coefficient", "OR"]])

# 输出危险因素（系数为正）和保护因素（系数为负）
print("\n危险因素（系数 > 0，增加死亡风险）：")
print(coef_df_sorted[coef_df_sorted["coefficient"] > 0][["feature", "medical_meaning", "coefficient", "OR"]])
print("\n保护因素（系数 < 0，降低死亡风险）：")
print(coef_df_sorted[coef_df_sorted["coefficient"] < 0][["feature", "medical_meaning", "coefficient", "OR"]])

# 绘制带医学含义的系数条形图
plt.figure(figsize=(10, 6))
# 按系数大小排序绘图，红色为正（危险），蓝色为负（保护）
colors = ["red" if c > 0 else "steelblue" for c in coef_df_sorted["coefficient"]]
ax = sns.barplot(
    data=coef_df_sorted,
    x="coefficient",
    y="feature",
    palette=colors
)

# 修改 y 轴标签为 "特征 (医学含义)"
y_labels = [f"{row['feature']} ({row['medical_meaning']})" for _, row in coef_df_sorted.iterrows()]
ax.set_yticklabels(y_labels)
plt.axvline(x=0, color="black", linestyle="--", linewidth=1)  # 加一条参考线
plt.xlabel("标准化系数 (Coefficient)")
plt.title(f"最佳模型（Logistic Regression）特征系数\n（红色=危险因素，蓝色=保护因素）")
plt.tight_layout()
plt.savefig("lr_coefficient_importance.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("逻辑回归系数重要性图已保存为 lr_coefficient_importance.png")

# 可选：输出前 3 个最重要的特征（供后续结论引用）
top3_lr = coef_df_sorted.head(3)
print("\n对 DEATH_EVENT 影响最大的前 3 个特征（基于 LR 系数）：")
print(top3_lr[["feature", "medical_meaning", "coefficient", "OR"]])


# =========================
# 12. t 检验：关键特征在死亡/未死亡患者组间的差异
# =========================
from scipy.stats import ttest_ind

t_test_features = [
    "age", "creatinine_phosphokinase", "ejection_fraction",
    "platelets", "serum_creatinine", "serum_sodium", "time"
]
group_dead = df_clean[df_clean["DEATH_EVENT"] == 1]
group_alive = df_clean[df_clean["DEATH_EVENT"] == 0]

t_test_results = []
for feature in t_test_features:
    dead_vals = group_dead[feature]
    alive_vals = group_alive[feature]
    t_stat, p_val = ttest_ind(dead_vals, alive_vals, equal_var=False)
    dead_mean_std = f"{dead_vals.mean():.2f} ± {dead_vals.std():.2f}"
    alive_mean_std = f"{alive_vals.mean():.2f} ± {alive_vals.std():.2f}"
    t_test_results.append({
        "特征": feature,
        "死亡组 (mean±std)": dead_mean_std,
        "未死亡组 (mean±std)": alive_mean_std,
        "t统计量": t_stat,
        "p值": p_val,
        "显著性 (p<0.05)": "是" if p_val < 0.05 else "否"
    })
t_test_df = pd.DataFrame(t_test_results)
print("\n=== t检验结果（死亡组 vs 未死亡组）===")
print(t_test_df.to_string(index=False))
t_test_df.to_csv("ttest_results.csv", index=False)
print("t检验结果已保存为 ttest_results.csv")

# =========================
# 13. 新患者示例输入与死亡概率预测（不含 time）
# =========================
new_patient = pd.DataFrame([{
    "age": 60,
    "anaemia": 1,
    "creatinine_phosphokinase": 582,
    "diabetes": 1,
    "ejection_fraction": 38,
    "high_blood_pressure": 0,
    "platelets": 263000,
    "serum_creatinine": 1.39,
    "serum_sodium": 136,
    "sex": 1,
    "smoking": 1
}])
print("\n新患者样本：")
print(new_patient)

# 使用 CV 选出的最佳模型
best_model = fitted_models[best_model_name]
death_prob = best_model.predict_proba(new_patient)[:, 1][0]
pred_label = int(death_prob >= 0.5)

print(f"当前选用的最佳模型（基于 CV AUC）：{best_model_name}")
print(f"预测死亡概率：{death_prob * 100:.2f}%")
print(f"预测类别（1=死亡事件，0=未发生死亡事件）：{pred_label}")
y_prob_best = best_model.predict_proba(X_test)[:,1]

# 校准曲线
prob_true, prob_pred = calibration_curve(
    y_test,
    y_prob_best,
    n_bins=10
)

brier = brier_score_loss(
    y_test,
    y_prob_best
)

print(
    f"Brier Score: {brier:.4f}"
)

# =========================
# 14. 绘制校准曲线
# =========================
plt.figure(figsize=(6,6))
plt.plot(
    prob_pred,
    prob_true,
    marker="o"
)
plt.plot(
    [0,1],
    [0,1],
    "--"
)
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.title("Calibration Curve")
plt.tight_layout()
plt.savefig(
    "calibration_curve.png",
    dpi=300
)
plt.close()

# =========================
# 15. Bootstrap AUC 置信区间
# =========================
def bootstrap_auc_ci(
    y_true,
    y_prob,
    n_bootstrap=1000,
    random_state=42
):
    rng = np.random.RandomState(random_state)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(
            0,
            len(y_true),
            len(y_true)
        )
        if len(np.unique(y_true.iloc[idx])) < 2:
            continue
        aucs.append(
            roc_auc_score(
                y_true.iloc[idx],
                y_prob[idx]
            )
        )
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    return lower, upper

lower, upper = bootstrap_auc_ci(
    y_test.reset_index(drop=True),
    y_prob_best
)

print(
    f"AUC 95% CI: [{lower:.3f}, {upper:.3f}]"
)

# # =========================
# # 16. 结果汇总与结论
# # =========================
# print("\n=== 最终模型评估汇总 ===")
# print(results_df.to_string(index=False))
#
# print("\n=== 结论提示 ===")
# print("1) 数据集无缺失值，仅报告异常值，未做任何截断（保留原始临床值）。")
# print("2) time 特征仅在 EDA 和异常值检测阶段保留，未用于模型训练与预测。")
# print("3) MLP 模型加入了 Early Stopping 和验证集，防止过拟合。")
# print("4) 所有模型均考虑了类别不平衡（LR: balanced, RF: balanced, XGB: scale_pos_weight）。")
# print("5) 模型选型基于 5 折分层交叉验证 AUC，避免单次拆分波动。")
# print("6) 特征重要性图已标注医学含义，便于临床解释。")