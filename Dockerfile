FROM  python:3.7
COPY . /usr/app/
EXPOSE 8501
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD streamlit run app.py --logger.level=debug 2> Generatedlogs/prediction_logs.log