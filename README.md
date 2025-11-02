# 🏥 HealthSolver v4.0

Clinical Decision Support System with 8 ML Models powered by XGBoost

## 🎯 Overview

HealthSolver is a predictive analytics platform that simultaneously evaluates 8 different clinical outcomes for patients, providing physicians with a comprehensive, multidimensional risk assessment.

## 🔬 ML Models

The application uses 8 XGBoost models trained on 50,000 synthetic samples:

1. **30-day mortality** - Immediate post-admission risk
2. **90-day mortality** - Short to medium-term prognosis  
3. **1-year mortality** - Long-term prognosis
4. **30-day readmission** - Hospital readmission probability
5. **Post-operative complications** - Surgical risk
6. **Cardiovascular events** - Heart attack, stroke, arrhythmias
7. **Functional decline** - Loss of autonomy
8. **Length of stay** - Estimated hospitalization days

## 📊 Features (58 inputs)

- Demographics: age, sex, BMI
- Vital signs: BP, HR, SpO2, temperature
- Lab results: HbA1c, eGFR, hemoglobin, creatinine, lipids, electrolytes
- Comorbidities: 13 chronic conditions
- Current condition: diagnosis, severity, acute/chronic phase

## ⚙️ Architecture

**Frontend**: HTML5, TailwindCSS, Vanilla JavaScript  
**Backend**: AWS Lambda (Python 3.11 container)  
**ML**: XGBoost, scikit-learn, 50k training samples  
**Infrastructure**: AWS (S3, Lambda, ECR, CodeBuild, CloudWatch)

## 🚀 Deployment

### Prerequisites
- AWS Account
- AWS CLI configured
- Docker installed
- Python 3.11+

### Backend Deployment

1. **Train Models**
\\\ash
pip install -r requirements.txt
python train_optimized.py
\\\

2. **Build Container**
\\\ash
docker build --platform linux/amd64 -t healthsolver-ml .
\\\

3. **Push to ECR**
\\\ash
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.eu-west-1.amazonaws.com
docker tag healthsolver-ml:latest YOUR_ACCOUNT.dkr.ecr.eu-west-1.amazonaws.com/healthsolver-ml:latest
docker push YOUR_ACCOUNT.dkr.ecr.eu-west-1.amazonaws.com/healthsolver-ml:latest
\\\

4. **Update Lambda**
\\\ash
aws lambda update-function-code \
    --function-name healthsolver-api \
    --image-uri YOUR_ACCOUNT.dkr.ecr.eu-west-1.amazonaws.com/healthsolver-ml:latest \
    --region eu-west-1
\\\

### Frontend Deployment

\\\ash
aws s3 cp full.html s3://your-bucket-name/ --region eu-west-1
\\\

## 📈 Model Performance

- **AUC**: 0.85-0.90 for classification models
- **MAE**: <2 days for length of stay regression
- **Training samples**: 50,000
- **Features**: 58
- **Inference time**: <2 seconds

## 🔐 Important Notes

⚠️ **For demonstration purposes only**  
- Models trained on synthetic data
- Not for clinical use without validation
- Requires regulatory approval for production

## 📄 License

MIT License - See LICENSE file

## 👤 Author

Riccardo Scaringi (@ricscar2570)

## 🔗 Links

- **Live Demo**: http://healthsolver-web-1761931563.s3-website-eu-west-1.amazonaws.com/full.html
- **LinkedIn**: https://www.linkedin.com/in/riccardoscaringi
- **Portfolio**: https://www.riccardoscaringi.eu/    <-- Update in progress

---

Built with ❤️ using AWS, XGBoost, and Python
