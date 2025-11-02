import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle

np.random.seed(42)
n_samples = 10000

data = {
    'age': np.random.randint(18, 95, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'height_cm': np.random.normal(170, 10, n_samples),
    'weight_kg': np.random.normal(75, 15, n_samples),
    'systolic_bp': np.random.normal(130, 20, n_samples),
    'diastolic_bp': np.random.normal(80, 12, n_samples),
    'heart_rate': np.random.normal(75, 15, n_samples),
    'oxygen_saturation': np.random.normal(96, 3, n_samples),
    'respiratory_rate': np.random.normal(16, 4, n_samples),
    'body_temperature': np.random.normal(36.8, 0.5, n_samples),
    'hemoglobin': np.random.normal(14, 2, n_samples),
    'wbc': np.random.normal(7, 2, n_samples),
    'platelet_count': np.random.normal(250, 80, n_samples),
    'fasting_glucose': np.random.normal(100, 30, n_samples),
    'hba1c': np.random.normal(6.0, 1.5, n_samples),
    'total_cholesterol': np.random.normal(200, 40, n_samples),
    'ldl_cholesterol': np.random.normal(120, 35, n_samples),
    'hdl_cholesterol': np.random.normal(50, 15, n_samples),
    'triglycerides': np.random.normal(150, 60, n_samples),
    'creatinine': np.random.normal(1.0, 0.4, n_samples),
    'egfr': np.random.normal(80, 25, n_samples),
    'bun': np.random.normal(18, 8, n_samples),
    'alt': np.random.normal(30, 15, n_samples),
    'ast': np.random.normal(28, 12, n_samples),
    'sodium': np.random.normal(140, 4, n_samples),
    'potassium': np.random.normal(4.0, 0.5, n_samples),
    'crp': np.random.normal(5, 8, n_samples),
    'has_hypertension': np.random.randint(0, 2, n_samples),
    'has_diabetes': np.random.randint(0, 2, n_samples),
    'has_atrial_fibrillation': np.random.randint(0, 2, n_samples),
    'has_heart_failure': np.random.randint(0, 2, n_samples),
    'has_coronary_artery_disease': np.random.randint(0, 2, n_samples),
    'has_stroke_history': np.random.randint(0, 2, n_samples),
    'has_chronic_kidney_disease': np.random.randint(0, 2, n_samples),
    'has_copd': np.random.randint(0, 2, n_samples),
    'has_asthma': np.random.randint(0, 2, n_samples),
    'has_cancer': np.random.randint(0, 2, n_samples),
    'has_liver_disease': np.random.randint(0, 2, n_samples),
    'has_depression': np.random.randint(0, 2, n_samples),
    'has_dementia': np.random.randint(0, 2, n_samples),
    'condition_severity': np.random.randint(1, 11, n_samples),
    'is_acute': np.random.randint(0, 2, n_samples),
}

df = pd.DataFrame(data)
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

mortality_risk = (
    (df['age'] > 75) * 0.15 +
    (df['systolic_bp'] > 160) * 0.10 +
    (df['heart_rate'] > 100) * 0.08 +
    (df['hba1c'] > 8.0) * 0.12 +
    (df['egfr'] < 60) * 0.15 +
    df['has_diabetes'] * 0.08 +
    df['has_hypertension'] * 0.05 +
    df['has_heart_failure'] * 0.20 +
    df['has_atrial_fibrillation'] * 0.12 +
    df['has_stroke_history'] * 0.15 +
    df['has_chronic_kidney_disease'] * 0.13 +
    df['has_coronary_artery_disease'] * 0.10 +
    df['has_copd'] * 0.09 +
    df['has_cancer'] * 0.18 +
    df['has_liver_disease'] * 0.14 +
    df['has_dementia'] * 0.11 +
    (df['condition_severity'] > 7) * 0.10 +
    (df['hemoglobin'] < 10) * 0.08 +
    (df['creatinine'] > 2.0) * 0.12 +
    np.random.normal(0, 0.1, n_samples)
)

df['mortality_30d'] = (mortality_risk > 0.5).astype(int)

feature_cols = [col for col in df.columns if col != 'mortality_30d']
X = df[feature_cols]
y = df['mortality_30d']

print(f'Dataset: {len(X)} samples, {len(feature_cols)} features')
print(f'Mortality: {y.sum()} ({y.mean()*100:.1f}%)')

model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
print('Training...')
model.fit(X, y)

with open('mortality_model_full.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model saved!')
feature_importance = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in feature_importance[:10]:
    print(f'  {feat}: {imp:.4f}')
