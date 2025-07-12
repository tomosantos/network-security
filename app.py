import os
import sys
import pymongo
import pandas as pd

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.main.utils import load_object
from src.utils.ml.model.estimator import NetworkModel

from src.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from src.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME
from src.constant.training_pipeline import MODEL_FILE_NAME, PREPROCESSING_OBJECT_FILE_NAME, FINAL_MODEL_DIR
from src.constant.training_pipeline import OUTPUT_FILE_NAME, OUTPUT_FILE_PATH

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv('MONGODB_URL_KEY')

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates


model_file_path = os.path.join(FINAL_MODEL_DIR, MODEL_FILE_NAME)
preprocessor_file_path = os.path.join(FINAL_MODEL_DIR, PREPROCESSING_OBJECT_FILE_NAME)
output_file_path = os.path.join(OUTPUT_FILE_PATH, OUTPUT_FILE_NAME)

app = FastAPI()
origins = ['*']
templates = Jinja2Templates(directory='./templates')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')


@app.get('/train')
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response('Training was successful')
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

@app.post('/predict')
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object(preprocessor_file_path)
        final_model = load_object(model_file_path)
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred

        df.to_csv(output_file_path)
        table_html = df.to_html(classes='table table-striped')

        return templates.TemplateResponse('table.html', {'request': request, 'table': table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
if __name__ == '__main__':
    app_run(app, host='0.0.0.0', port=8080)