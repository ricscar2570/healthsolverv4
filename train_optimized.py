import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error

np.random.seed(42)
n_samples = 50000  # 5x più dati per modelli migliori

print('📊 Creating enhanced dataset with 50,000 samples...')

# Dataset più realistico con correlazioni
age = np.random.randint(18, 95, n_samples)
sex = np.random.randint(0, 2, n_samples)

# Correlazioni realistiche
has_hypertension = ((age > 50) * 0.4 + np.random.random(n_samples) > 0.7).astype(int)
has_diabetes = ((age > 55) * 0.3 + has_hypertension * 0.2 + np.random.random(n_samples) > 0.75).astype(int)
has_cad = ((age > 60) * 0.35 + has_diabetes * 0.25 + has_hypertension * 0.20 + np.random.random(n_samples) > 0.8).astype(int)
has_heart_failure = ((age > 65) * 0.3 + has_cad * 0.4 + has_hypertension * 0.25 + np.random.random(n_samples) > 0.85).astype(int)
has_atrial_fib = ((age > 70) * 0.35 + has_heart_failure * 0.3 + np.random.random(n_samples) > 0.85).astype(int)
has_ckd = ((age > 60) * 0.3 + has_diabetes * 0.35 + has_hypertension * 0.25 + np.random.random(n_samples) > 0.8).astype(int)
has_copd = ((age > 55) * 0.25 + (sex == 1) * 0.15 + np.random.random(n_samples) > 0.85).astype(int)
has_stroke = ((age > 65) * 0.3 + has_atrial_fib * 0.4 + has_hypertension * 0.2 + np.random.random(n_samples) > 0.88).astype(int)
has_cancer = ((age > 60) * 0.2 + np.random.random(n_samples) > 0.90).astype(int)
has_liver = ((age > 50) * 0.15 + np.random.random(n_samples) > 0.92).astype(int)
has_asthma = np.random.randint(0, 2, n_samples) * (np.random.random(n_samples) > 0.90)
has_depression = ((age > 40) * 0.2 + np.random.random(n_samples) > 0.80).astype(int)
has_dementia = ((age > 75) * 0.35 + np.random.random(n_samples) > 0.85).astype(int)

# Parametri vitali correlati
systolic_bp = 110 + age * 0.5 + has_hypertension * 25 + np.random.normal(0, 15, n_samples)
diastolic_bp = 70 + age * 0.2 + has_hypertension * 12 + np.random.normal(0, 10, n_samples)
heart_rate = 70 + has_heart_failure * 15 + has_atrial_fib * 20 + np.random.normal(0, 12, n_samples)
oxygen_sat = 98 - has_copd * 4 - has_heart_failure * 3 + np.random.normal(0, 2, n_samples)
resp_rate = 14 + has_copd * 4 + has_heart_failure * 3 + np.random.normal(0, 3, n_samples)
body_temp = 36.8 + np.random.normal(0, 0.4, n_samples)

# Lab correlati
hba1c = 5.5 + has_diabetes * 2.0 + np.random.normal(0, 1.0, n_samples)
egfr = 95 - age * 0.6 - has_ckd * 35 - has_diabetes * 15 + np.random.normal(0, 15, n_samples)
creatinine = 1.0 + (has_ckd * 0.8) + ((95 - egfr) / 50) + np.random.normal(0, 0.3, n_samples)
hemoglobin = 14 - (has_ckd * 2) - (has_cancer * 1.5) + (sex * 1.5) + np.random.normal(0, 1.5, n_samples)
wbc = 7 + (has_cancer * 2) + np.random.normal(0, 2, n_samples)
platelets = 250 + np.random.normal(0, 60, n_samples)
glucose = 95 + has_diabetes * 45 + np.random.normal(0, 25, n_samples)
cholesterol = 190 + age * 0.3 + np.random.normal(0, 35, n_samples)
ldl = 120 + has_diabetes * 20 + np.random.normal(0, 30, n_samples)
hdl = 50 - has_diabetes * 8 + (sex == 0) * 10 + np.random.normal(0, 12, n_samples)
triglycerides = 140 + has_diabetes * 60 + np.random.normal(0, 50, n_samples)
bun = 18 + (has_ckd * 15) + np.random.normal(0, 7, n_samples)
alt = 28 + has_liver * 40 + np.random.normal(0, 15, n_samples)
ast = 25 + has_liver * 45 + np.random.normal(0, 12, n_samples)
sodium = 140 + np.random.normal(0, 3, n_samples)
potassium = 4.0 + (has_ckd * 0.5) + np.random.normal(0, 0.4, n_samples)
crp = 3 + has_copd * 5 + has_heart_failure * 4 + np.random.exponential(3, n_samples)

height_cm = 170 + (sex * 10) + np.random.normal(0, 8, n_samples)
weight_kg = 75 + (age - 40) * 0.15 + (sex * 8) + np.random.normal(0, 12, n_samples)
bmi = weight_kg / ((height_cm / 100) ** 2)

condition_severity = np.random.randint(1, 11, n_samples)
is_acute = np.random.randint(0, 2, n_samples)

data = {
    'age': age, 'sex': sex, 'height_cm': height_cm, 'weight_kg': weight_kg,
    'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
    'heart_rate': heart_rate, 'oxygen_saturation': oxygen_sat,
    'respiratory_rate': resp_rate, 'body_temperature': body_temp,
    'hemoglobin': hemoglobin, 'wbc': wbc, 'platelet_count': platelets,
    'fasting_glucose': glucose, 'hba1c': hba1c,
    'total_cholesterol': cholesterol, 'ldl_cholesterol': ldl,
    'hdl_cholesterol': hdl, 'triglycerides': triglycerides,
    'creatinine': creatinine, 'egfr': egfr, 'bun': bun,
    'alt': alt, 'ast': ast, 'sodium': sodium, 'potassium': potassium, 'crp': crp,
    'has_hypertension': has_hypertension, 'has_diabetes': has_diabetes,
    'has_atrial_fibrillation': has_atrial_fib, 'has_heart_failure': has_heart_failure,
    'has_coronary_artery_disease': has_cad, 'has_stroke_history': has_stroke,
    'has_chronic_kidney_disease': has_ckd, 'has_copd': has_copd,
    'has_asthma': has_asthma, 'has_cancer': has_cancer,
    'has_liver_disease': has_liver, 'has_depression': has_depression,
    'has_dementia': has_dementia, 'condition_severity': condition_severity,
    'is_acute': is_acute, 'bmi': bmi
}

df = pd.DataFrame(data)
X = df

print(f'✅ Dataset created: {len(df)} samples, {len(df.columns)} features\n')
print('🔬 Training 7 optimized models with validation...\n')

# Parametri ottimizzati per tutti i modelli
params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# 1. MORTALITÀ 90 GIORNI
print('1/7 📊 Mortality 90d (optimized)...')
mort_90d = (
    (age > 75) * 0.18 + (age > 85) * 0.15 +
    (systolic_bp > 160) * 0.12 + (systolic_bp > 180) * 0.10 +
    (heart_rate > 100) * 0.10 + (heart_rate > 120) * 0.08 +
    (hba1c > 8.0) * 0.14 + (hba1c > 9.5) * 0.10 +
    (egfr < 60) * 0.18 + (egfr < 30) * 0.20 +
    has_diabetes * 0.10 + has_heart_failure * 0.25 +
    has_atrial_fib * 0.15 + has_stroke * 0.18 +
    has_cancer * 0.22 + has_copd * 0.11 +
    has_liver * 0.16 + has_dementia * 0.12 +
    (condition_severity > 7) * 0.12 + (condition_severity > 9) * 0.08 +
    (hemoglobin < 10) * 0.12 + (creatinine > 2.0) * 0.14 +
    (oxygen_sat < 92) * 0.15 + (bmi < 18) * 0.08 +
    np.random.normal(0, 0.10, n_samples)
)
y1 = (mort_90d > 0.55).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y1, test_size=0.2, random_state=42)
m1 = XGBClassifier(**params)
m1.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
auc1 = roc_auc_score(y_te, m1.predict_proba(X_te)[:, 1])
with open('mortality_90d_model.pkl', 'wb') as f: pickle.dump(m1, f)
print(f'   ✅ Prevalence: {y1.mean()*100:.1f}% | AUC: {auc1:.3f}')

# 2. MORTALITÀ 1 ANNO
print('2/7 📊 Mortality 1yr (optimized)...')
mort_1yr = mort_90d * 1.25 + (
    (age > 80) * 0.12 + has_heart_failure * 0.15 +
    has_cancer * 0.18 + has_dementia * 0.14 +
    (egfr < 45) * 0.15 + has_copd * 0.10 +
    np.random.normal(0, 0.12, n_samples)
)
y2 = (mort_1yr > 0.65).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y2, test_size=0.2, random_state=43)
m2 = XGBClassifier(**params)
m2.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
auc2 = roc_auc_score(y_te, m2.predict_proba(X_te)[:, 1])
with open('mortality_1yr_model.pkl', 'wb') as f: pickle.dump(m2, f)
print(f'   ✅ Prevalence: {y2.mean()*100:.1f}% | AUC: {auc2:.3f}')

# 3. RIAMMISSIONE 30 GIORNI
print('3/7 📊 Readmission 30d (optimized)...')
readmit = (
    (age > 70) * 0.12 + (age > 80) * 0.08 +
    has_heart_failure * 0.25 + has_copd * 0.18 +
    has_diabetes * 0.15 + has_ckd * 0.20 +
    (egfr < 45) * 0.16 + (egfr < 30) * 0.12 +
    (condition_severity > 6) * 0.14 + (condition_severity > 8) * 0.10 +
    has_depression * 0.10 + has_dementia * 0.12 +
    (sodium < 135) * 0.12 + (sodium < 130) * 0.10 +
    (hemoglobin < 11) * 0.10 + (hemoglobin < 9) * 0.08 +
    is_acute * 0.08 + (bmi > 35) * 0.07 +
    np.random.normal(0, 0.12, n_samples)
)
y3 = (readmit > 0.45).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y3, test_size=0.2, random_state=44)
m3 = XGBClassifier(**params)
m3.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
auc3 = roc_auc_score(y_te, m3.predict_proba(X_te)[:, 1])
with open('readmission_30d_model.pkl', 'wb') as f: pickle.dump(m3, f)
print(f'   ✅ Prevalence: {y3.mean()*100:.1f}% | AUC: {auc3:.3f}')

# 4. COMPLICANZE POST-OPERATORIE
print('4/7 📊 Post-op complications (optimized)...')
complications = (
    (age > 70) * 0.15 + (age > 80) * 0.12 +
    (bmi > 35) * 0.12 + (bmi > 40) * 0.10 +
    (bmi < 18) * 0.08 + has_diabetes * 0.18 +
    (hba1c > 7.5) * 0.14 + (hba1c > 9.0) * 0.10 +
    has_copd * 0.16 + has_heart_failure * 0.20 +
    (creatinine > 1.5) * 0.13 + (creatinine > 2.0) * 0.12 +
    (hemoglobin < 10) * 0.15 + (hemoglobin < 8) * 0.12 +
    (egfr < 60) * 0.11 + has_liver * 0.16 +
    (condition_severity > 7) * 0.12 + is_acute * 0.10 +
    (oxygen_sat < 94) * 0.11 + has_cancer * 0.14 +
    np.random.normal(0, 0.11, n_samples)
)
y4 = (complications > 0.50).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y4, test_size=0.2, random_state=45)
m4 = XGBClassifier(**params)
m4.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
auc4 = roc_auc_score(y_te, m4.predict_proba(X_te)[:, 1])
with open('complications_model.pkl', 'wb') as f: pickle.dump(m4, f)
print(f'   ✅ Prevalence: {y4.mean()*100:.1f}% | AUC: {auc4:.3f}')

# 5. EVENTI CARDIOVASCOLARI
print('5/7 📊 CV events (optimized)...')
cv_events = (
    (age > 65) * 0.14 + (age > 75) * 0.12 +
    has_hypertension * 0.18 + (systolic_bp > 160) * 0.15 +
    (systolic_bp > 180) * 0.12 + has_diabetes * 0.16 +
    (ldl > 160) * 0.14 + (ldl > 190) * 0.10 +
    (hdl < 40) * 0.10 + (hdl < 35) * 0.08 +
    has_cad * 0.25 + has_atrial_fib * 0.20 +
    has_stroke * 0.22 + has_heart_failure * 0.18 +
    (crp > 10) * 0.12 + (crp > 20) * 0.08 +
    (triglycerides > 200) * 0.10 + has_ckd * 0.14 +
    (condition_severity > 6) * 0.09 + (sex == 1) * 0.05 +
    np.random.normal(0, 0.11, n_samples)
)
y5 = (cv_events > 0.52).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y5, test_size=0.2, random_state=46)
m5 = XGBClassifier(**params)
m5.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
auc5 = roc_auc_score(y_te, m5.predict_proba(X_te)[:, 1])
with open('cv_events_model.pkl', 'wb') as f: pickle.dump(m5, f)
print(f'   ✅ Prevalence: {y5.mean()*100:.1f}% | AUC: {auc5:.3f}')

# 6. DEGENERAZIONE FUNZIONALE
print('6/7 📊 Functional decline (optimized)...')
decline = (
    (age > 75) * 0.20 + (age > 85) * 0.15 +
    has_dementia * 0.25 + has_stroke * 0.22 +
    (hemoglobin < 11) * 0.12 + (hemoglobin < 9) * 0.10 +
    has_heart_failure * 0.18 + has_copd * 0.15 +
    (condition_severity > 7) * 0.14 + (condition_severity > 9) * 0.10 +
    has_depression * 0.13 + has_cancer * 0.16 +
    (bmi < 20) * 0.10 + (bmi < 18) * 0.08 +
    (egfr < 45) * 0.12 + has_ckd * 0.11 +
    (oxygen_sat < 94) * 0.10 + is_acute * 0.08 +
    np.random.normal(0, 0.11, n_samples)
)
y6 = (decline > 0.48).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y6, test_size=0.2, random_state=47)
m6 = XGBClassifier(**params)
m6.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
auc6 = roc_auc_score(y_te, m6.predict_proba(X_te)[:, 1])
with open('functional_decline_model.pkl', 'wb') as f: pickle.dump(m6, f)
print(f'   ✅ Prevalence: {y6.mean()*100:.1f}% | AUC: {auc6:.3f}')

# 7. DURATA RICOVERO - REGRESSIONE
print('7/7 📊 Length of stay (regression, optimized)...')
los = (
    3.0 + (age - 50) * 0.08 + (age > 75) * 2 +
    condition_severity * 1.2 + (condition_severity > 7) * 2 +
    has_heart_failure * 4 + has_copd * 3 +
    has_diabetes * 2 + is_acute * 2 +
    (hemoglobin < 10) * 3 + (egfr < 45) * 3 +
    has_cancer * 3.5 + has_stroke * 3 +
    (oxygen_sat < 92) * 2.5 + has_dementia * 2 +
    (bmi > 35) * 1.5 + has_liver * 2.5 +
    np.random.normal(0, 1.5, n_samples)
).clip(1, 30)
X_tr, X_te, y_tr, y_te = train_test_split(X, los, test_size=0.2, random_state=48)
m7 = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=48)
m7.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
mae = mean_absolute_error(y_te, m7.predict(X_te))
with open('length_of_stay_model.pkl', 'wb') as f: pickle.dump(m7, f)
print(f'   ✅ Mean: {los.mean():.1f} days | MAE: {mae:.2f} days')

print('\n🎉 ALL 7 OPTIMIZED MODELS TRAINED!')
print(f'\n📈 Average AUC: {(auc1+auc2+auc3+auc4+auc5+auc6)/6:.3f}')
print('\nModels saved with enhanced performance!')
