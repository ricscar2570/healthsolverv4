FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install -r requirements.txt --target /var/task

COPY lambda_handler.py /var/task/
COPY mortality_model_full.pkl /var/task/
COPY mortality_90d_model.pkl /var/task/
COPY mortality_1yr_model.pkl /var/task/
COPY readmission_30d_model.pkl /var/task/
COPY complications_model.pkl /var/task/
COPY cv_events_model.pkl /var/task/
COPY functional_decline_model.pkl /var/task/
COPY length_of_stay_model.pkl /var/task/

CMD ["lambda_handler.lambda_handler"]