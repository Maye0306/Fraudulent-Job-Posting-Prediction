import pandas as pd
import numpy as np
import re
import warnings  # 正确导入warnings模块
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# 过滤特定警告（直接使用警告类别）
warnings.filterwarnings("ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names"
)

# 配置参数
TEXT_MAX_FEATURES = 1500
N_FOLDS = 5
CLASS_WEIGHT = {0: 1, 1: 15}

def text_cleaner(text):
    """文本清洗增强版"""
    text = re.sub(r'[^\w\s]', '', str(text))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text[:500]

def preprocess_data(df):
    """数据预处理优化"""
    # 删除无关字段
    df = df.drop(['job_id', 'salary_range'], axis=1, errors='ignore')

    # 处理文本字段
    text_cols = ['company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        df[col] = df[col].fillna('').apply(text_cleaner)
        df[f'{col}_wc'] = df[col].apply(lambda x: len(x.split()))
        df[f'{col}_url'] = df[col].str.contains('http').astype(int)

    # 生成组合文本
    df['combined_text'] = df[text_cols].apply(lambda x: ' '.join(x), axis=1)

    # 数值特征
    df['has_salary'] = (~df['salary_range'].isna()).astype(int) if 'salary_range' in df else 0
    df['title_len'] = df['title'].apply(lambda x: len(str(x)))

    # 分类特征处理
    cat_cols = ['employment_type', 'required_experience', 'required_education',
                'industry', 'function', 'department', 'title', 'location']
    for col in cat_cols:
        df[col] = df[col].fillna('missing').astype(str).str[:20].str.replace(' ', '_')

    # 删除原始文本字段
    df = df.drop(text_cols, axis=1)

    return df

# 加载数据
train = preprocess_data(pd.read_csv('fake_job_postings_train.csv'))
test = preprocess_data(pd.read_csv('fake_job_postings_test.csv'))

# 特征定义
text_feature = 'combined_text'
num_features = [
    'telecommuting', 'has_company_logo', 'has_questions',
    'has_salary', 'title_len',
    'company_profile_wc', 'description_wc',
    'requirements_wc', 'benefits_wc',
    'company_profile_url', 'description_url',
    'requirements_url', 'benefits_url'
]
cat_features = [
    'employment_type', 'required_experience', 'required_education',
    'industry', 'function', 'department', 'title', 'location'
]

# 构建预处理Pipeline
preprocessor = ColumnTransformer([
    ('text', TfidfVectorizer(
        max_features=TEXT_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english'
    ), text_feature),
    ('cat', OneHotEncoder(
        max_categories=20,
        handle_unknown='infrequent_if_exist',
        sparse_output=False
    ), cat_features),
    ('num', 'passthrough', num_features)
], remainder='drop')

# 优化模型参数
model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=6,
    num_leaves=40,
    reg_alpha=0.2,
    reg_lambda=0.2,
    class_weight=CLASS_WEIGHT,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    verbosity=-1
)

# 构建完整Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# 交叉验证训练
X = train.drop('fraudulent', axis=1)
y = train['fraudulent']
test_preds = np.zeros(test.shape[0])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/{N_FOLDS}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict_proba(X_val)[:, 1]
    print(f"Validation AUC: {roc_auc_score(y_val, val_pred):.4f}")

    test_preds += pipeline.predict_proba(test)[:, 1] / N_FOLDS

# 生成提交文件
submission = pd.DataFrame({
    'job_id': pd.read_csv('fake_job_postings_test.csv')['job_id'],
    'pred': test_preds.round(4)
})
submission.to_csv('submission.csv', index=False)
print("预测结果已保存至 submission.csv")