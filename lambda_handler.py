import json
import pickle
import numpy as np
from datetime import datetime

# Carica TUTTI i modelli
with open('mortality_model_full.pkl', 'rb') as f:
    model_30d = pickle.load(f)
with open('mortality_90d_model.pkl', 'rb') as f:
    model_90d = pickle.load(f)
with open('mortality_1yr_model.pkl', 'rb') as f:
    model_1yr = pickle.load(f)
with open('readmission_30d_model.pkl', 'rb') as f:
    model_readmit = pickle.load(f)
with open('complications_model.pkl', 'rb') as f:
    model_complications = pickle.load(f)
with open('cv_events_model.pkl', 'rb') as f:
    model_cv = pickle.load(f)
with open('functional_decline_model.pkl', 'rb') as f:
    model_decline = pickle.load(f)
with open('length_of_stay_model.pkl', 'rb') as f:
    model_los = pickle.load(f)

def extract_features(body):
    d = body.get('demographics', {})
    v = body.get('vital_signs', {})
    l = body.get('lab_results', {})
    c = body.get('comorbidities', {})
    cc = body.get('current_condition', {})
    
    height = v.get('height_cm', 170)
    weight = v.get('weight_kg', 75)
    bmi = weight / ((height / 100) ** 2) if height > 0 else 25
    
    features = [
        d.get('age', 0), 1 if d.get('sex') == 'male' else 0, height, weight,
        v.get('systolic_bp', 0), v.get('diastolic_bp', 0), v.get('heart_rate', 0),
        v.get('oxygen_saturation', 0), v.get('respiratory_rate', 0), v.get('body_temperature', 0),
        l.get('hemoglobin', 0), l.get('wbc', 0), l.get('platelet_count', 0),
        l.get('fasting_glucose', 0), l.get('hba1c', 0), l.get('total_cholesterol', 0),
        l.get('ldl_cholesterol', 0), l.get('hdl_cholesterol', 0), l.get('triglycerides', 0),
        l.get('creatinine', 0), l.get('egfr', 0), l.get('bun', 0),
        l.get('alt', 0), l.get('ast', 0), l.get('sodium', 0), l.get('potassium', 0), l.get('crp', 0),
        int(c.get('has_hypertension', False)), int(c.get('has_diabetes', False)),
        int(c.get('has_atrial_fibrillation', False)), int(c.get('has_heart_failure', False)),
        int(c.get('has_coronary_artery_disease', False)), int(c.get('has_stroke_history', False)),
        int(c.get('has_chronic_kidney_disease', False)), int(c.get('has_copd', False)),
        int(c.get('has_asthma', False)), int(c.get('has_cancer', False)),
        int(c.get('has_liver_disease', False)), int(c.get('has_depression', False)),
        int(c.get('has_dementia', False)), cc.get('condition_severity', 5),
        1 if cc.get('acute_or_chronic') == 'acute' else 0, bmi
    ]
    return features

def lambda_handler(event, context):
    headers = {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*', 
               'Access-Control-Allow-Methods': 'GET,POST,OPTIONS', 'Access-Control-Allow-Headers': '*'}
    
    if event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS':
        return {'statusCode': 200, 'headers': headers, 'body': ''}
    
    path = event.get('rawPath', '/')
    
    if path == '/health':
        return {'statusCode': 200, 'headers': headers, 'body': json.dumps({
            'status': 'healthy', 'service': 'HealthSolver ML - Complete Suite', 'version': '4.0.0',
            'ml_enabled': True, 'models': 8,
            'models_list': ['Mortality 30d', 'Mortality 90d', 'Mortality 1yr', 'Readmission 30d',
                          'Post-op Complications', 'CV Events', 'Functional Decline', 'Length of Stay'],
            'timestamp': datetime.utcnow().isoformat()
        })}
    
    if path == '/analyze':
        try:
            body = json.loads(event.get('body', '{}'))
            features = extract_features(body)
            
            # Predizioni - converti numpy float32 a Python float
            prob_30d = model_30d.predict_proba([features])[0]
            prob_90d = model_90d.predict_proba([features])[0]
            prob_1yr = model_1yr.predict_proba([features])[0]
            prob_readmit = model_readmit.predict_proba([features])[0]
            prob_complications = model_complications.predict_proba([features])[0]
            prob_cv = model_cv.predict_proba([features])[0]
            prob_decline = model_decline.predict_proba([features])[0]
            los_days = float(model_los.predict([features])[0])
            
            def get_category(prob):
                return 'ALTO' if prob > 0.5 else 'MODERATO' if prob > 0.3 else 'BASSO'
            
            return {'statusCode': 200, 'headers': headers, 'body': json.dumps({
                'patient_id': body.get('patient_id'),
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': 'OK',
                'ml_predictions': {
                    'mortality_30d': {
                        'probability': round(float(prob_30d[1]) * 100, 1),
                        'category': get_category(prob_30d[1]),
                        'confidence': round(float(max(prob_30d)) * 100, 1)
                    },
                    'mortality_90d': {
                        'probability': round(float(prob_90d[1]) * 100, 1),
                        'category': get_category(prob_90d[1]),
                        'confidence': round(float(max(prob_90d)) * 100, 1)
                    },
                    'mortality_1yr': {
                        'probability': round(float(prob_1yr[1]) * 100, 1),
                        'category': get_category(prob_1yr[1]),
                        'confidence': round(float(max(prob_1yr)) * 100, 1)
                    },
                    'readmission_30d': {
                        'probability': round(float(prob_readmit[1]) * 100, 1),
                        'category': get_category(prob_readmit[1]),
                        'confidence': round(float(max(prob_readmit)) * 100, 1)
                    },
                    'complications': {
                        'probability': round(float(prob_complications[1]) * 100, 1),
                        'category': get_category(prob_complications[1]),
                        'confidence': round(float(max(prob_complications)) * 100, 1)
                    },
                    'cv_events': {
                        'probability': round(float(prob_cv[1]) * 100, 1),
                        'category': get_category(prob_cv[1]),
                        'confidence': round(float(max(prob_cv)) * 100, 1)
                    },
                    'functional_decline': {
                        'probability': round(float(prob_decline[1]) * 100, 1),
                        'category': get_category(prob_decline[1]),
                        'confidence': round(float(max(prob_decline)) * 100, 1)
                    },
                    'length_of_stay': {
                        'days': round(los_days, 1),
                        'category': 'LUNGO' if los_days > 10 else 'MEDIO' if los_days > 5 else 'BREVE'
                    },
                    'model_version': '4.0.0'
                },
                'features': {'count': len(features), 'data': {
                    'age': features[0], 'sex': features[1], 'sbp': features[4], 'hr': features[6],
                    'hba1c': features[14], 'egfr': features[20], 'diabetes': features[28],
                    'hypertension': features[27], 'atrial_fib': features[29], 'heart_failure': features[30],
                    'cad': features[31], 'stroke': features[32], 'ckd': features[33],
                    'copd': features[34], 'cancer': features[36]
                }},
                'clinical_warnings': {'count': 0, 'warnings': []}
            })}
        except Exception as e:
            import traceback
            return {'statusCode': 500, 'headers': headers,
                   'body': json.dumps({'error': str(e), 'trace': traceback.format_exc()})}
    
    return {'statusCode': 404, 'headers': headers, 'body': json.dumps({'error': 'Not found'})}
