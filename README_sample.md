# Unified ETL Data warehousing MLops Pipeline
This repository is part of an **MLOps workflow**, showcasing a complete pipeline that integrates **data engineering** and **machine learning** for real-time **credit risk prediction**

You'll learn key concepts like **data ingestion**, **ETL (Extract, Transform, Load)**, **data warehousing**, **ML model deployment**, and **online prediction** using **Kubeflow** and various **GCP services**.

We use the [German Credit Risk dataset](https://www.kaggle.com/uciml/german-credit) and simulate a **streaming data source** that emits customer details in real time. The pipeline is designed to predict customer credit risk, covering the full lifecycle — from batch ingestion and processing, to model training, deployment, and online inference—detailed in the architecture diagram below.

![ML Ops Architecture_new](https://github.com/user-attachments/assets/3bebdae3-4a8a-4c43-a3f4-88b96eb2bfcd)


1. **Ingest Data from the source in batch format** 
    - Export historical customer data in CSV format and store it in **Google Cloud Storage (GCS)**.
    - [Go to Code explanation](#1-reading-the-data)
      
2. **Create and run a Batch Processing Dataflow ETL Job** [1]
    - Use **Apache Beam with Dataflow** to read, clean, and transform batch data.
    - Ensure proper schema alignment and data quality checks
    - [Go to Code Explanation](#2-create-batch-dataflow-job)
      
3. **Load transformed data into BigQuery**
    - Write the cleaned and enriched data to a **BigQuery table** for downstream analytics and ML use
    - [Go to Code Explanation](#3-inserting-data-in-bigquery)
      
4. **Create and run Kubeflow pipeline** [2]
    - Develop a **Vertex AI Kubeflow pipeline** that:
        - Prepares features
        - Performs Hyperparmeter Tuning
        - Performs Model training
        - Deploys it to a **Vertex AI Model Registry**
        - Creates Endpoints for online Prediction
        - [Go to Code Explanation](#4-create-kubeflow-pipeline)
          
5. **Ingest streaming Data from a the sources**
    - Enable real-time ingestion from a live customer data stream
    - [Go to Code Explanation](#5-reading-data-from-pub-sub) 
      
6. **Create and Run a Streaming Pipeline** [3]
    - Build a **Streaming Apache Beam pipeline** that reads, cleans and transforms input data.
    - [Go to Code Explanation](#6-create-streaming-dataflow-job)
      
7. **Perform online prediction using the deployed Vertex AI Endpoint**
    - Send transformed data to the deployed **ML model endpoint** for inference
    - [Go to Code Explanation](#7-online-prediction-using-vertex-ai-endpoint)
      
8. **Ingest output Data in Bigquery Table**
    - Store the prediction results in a **BigQuery table** for monitoring, analytics, or alerting.
    - [Go to Code Explanation](#8-inserting-data-in-bigquery)

9. **Monitor model predictions for drift or anomalies**
    - Analyze the prediction results stored in BigQuery for signs of data drift, class imbalance, or concept drift.
    - [Go to Code Explanation](#9-model-monitoring-using-vertex-ai)

10. **Trigger Cloud Alert on threshold breach**
    - When drift or anomaly is detected, create a alerting policy in **Cloud Alerting**.
    - [Go to Code Explanation](#10-triggering-cloud-alerts-on-threshold-breach)
    
11. **Create a log-based alert**
    - Threshold breach in **Cloud Alerting** triggers an incident and a message to **Pub Sub**.
    - The Message in **Pub sub** initiates a **Cloud Run Functions** service that start retraining pipeline.
    - [Go to Code Explanation](#11-handling-alerts-using-cloud-run-functions)
   
12. **Initiate automated retraining via Cloud Run Functions**
    - The **Cloud Run Functions** starts the Vertex AI Pipeline to retrain the model.
    - [Go to Code Explanation](#12-automated-model-retraining-via-cloud-run-functions)

***Optional: Infrastructure Setup using Terraform***

If you want to set up reproducible infrastructure for the required GCP components (buckets, Pub/Sub topics, BigQuery datasets, Artifact Registry, Vertex AI Workbench, Cloud Functions, etc.), you can provision it automatically using my Terraform repository:

**[GCP Infra for Data & AI using Terraform](https://github.com/adityasolanki205/GCP-Infra-for-Data-AI-using-Terraform)**

This step is **optional**.  
- If you already have your infrastructure set up, you can skip this.  
- If not, you can use this repository to provision all necessary components before executing the pipeline.

Reference:  
[1]: [Batch Dataflow Pipeline](https://github.com/adityasolanki205/Batch-Processing-Pipeline-using-DataFlow)  
[2]: [Kubeflow Pipeline](https://github.com/adityasolanki205/ML-Pipeline-using-Kubeflow)  
[3]: [Streaming Dataflow Pipeline](https://github.com/adityasolanki205/ML-Streaming-pipeline-using-Dataflow)

## Motivation
For the last few years, I have been part of a great learning curve wherein I have upskilled myself to move into a Machine Learning and Cloud Computing. This project was practice project for all the learnings I have had. This is first of the many more to come. 
 

## Libraries/frameworks used

<b>Built with</b>
- [Anaconda](https://www.anaconda.com/)
- [Python](https://www.python.org/)
- [Vertex AI](https://cloud.google.com/vertex-ai?hl=en)
- [Google Cloud Storage](https://cloud.google.com/storage)
- [Artifact Registry](https://cloud.google.com/artifact-registry/docs)
- [Cloud Build](https://cloud.google.com/build/docs)
- [Vertex AI Workbench](https://cloud.google.com/vertex-ai-notebooks?hl=en)
- [Vertex AI Model Registry](https://cloud.google.com/vertex-ai/docs/model-registry/introduction)
- [Vertex AI Online Prediction](https://cloud.google.com/vertex-ai/docs/predictions/get-predictions)
- [Vertex AI Metadata](https://cloud.google.com/vertex-ai/docs/ml-metadata/introduction)
- [Pub Sub](https://cloud.google.com/pubsub/docs)
- [Cloud Alerting](https://cloud.google.com/monitoring/alerts)
- [Cloud Run Functions](https://cloud.google.com/functions/docs)
- [Apache Beam](https://beam.apache.org/documentation/programming-guide/)
- [Google DataFlow](https://cloud.google.com/dataflow)

## Cloning Repository

```bash
    # clone this repo:
    git clone https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline.git
```

## Initial Setup

Below are the steps to setup the enviroment and run the codes:
 
1. **Setup**: First we will have to setup free google cloud account which can be done [here](https://cloud.google.com/free). Then we need to Download the data from [German Credit Risk](https://www.kaggle.com/uciml/german-credit). Also present in the repository [here](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/german_data.csv)

2. **Creating a Vertex AI Workbench, Cloud Storage bucket and SDK**: Here will we will create workbench to run the Kubeflow pipeline and S3 bucket to be used in the process.

    - Goto to Vertex AI workbench
    - Select Instances, Click on Create New and create the instance in asia-south1 with default settings
    - After the instance becomes active, click on Juptyter Labs. Open a terminal and run the below command.
    ```bash
       git clone https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline.git
       cd Unified-ETL-DWH-MLOps-Pipeline
    ```
    - Similarly copy the commands in Cloud sdk as well.
    ```bash
        git clone https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline.git
        cd Unified-ETL-DWH-MLOps-Pipeline
    ```
    - Goto to Storage Bucket
    - Click on create new and create a bucket with default setting in asia-south1 with the name 'demo-bucket-kfl'
    - Copy the file using 'gsutil cp german_data.csv gs://demo_bucket_kfl/'

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/89739148-baa3-4d95-9371-b2bab4ae4ead

3. **Creating a Artifact Registry**: We will now create a Repository for our Docker Image to be stored. Process is provded below.

    - Goto to Artifact registry.
    - Click on create Repository, use default setting to create a Docker Repository in asia-south1 and the name
      'kubeflow-pipelines'

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/76903e65-c08c-46f9-b86b-34de96268290

4. **Creating the Docker Image**: After creating the repository we will create the docker Image for Kubeflow Components. This will also install all the required libraries:

   - To create this image we go back to workbench.
   - Now we run the docker_build. This file contains all the commands to create the image. It also contains requirements.txt file to install all the dependancies.
     
    requirements.txt
    ```text
        pandas
        numpy
        scikit-learn
        joblib
        Cython
        hyperopt
        kfp
        db-dtypes
        
        # Google Cloud libraries
        google-cloud-aiplatform
        google-cloud-storage
        google-cloud-pubsub
        google-cloud-bigquery
        google-cloud-bigquery-storage
        googleapis-common-protos
    ```
    
    docker_build.sh
    ```bash
        FROM gcr.io/deeplearning-platform-release/base-cpu
        
        WORKDIR /
        COPY training_pipeline.py /
        COPY requirements.txt /
        COPY ./src/ /src
        RUN pip install --upgrade pip && pip install -r requirements.txt
    ```
    - To create the image, we run the command below

    ```bash
       bash docker_build.sh
    ```
##### Related codes
1. [dockerfile](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/Dockerfile)
2. [requirements.txt](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/requirements.txt)
3. [docker_build.sh](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/docker_build.sh)

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/0ccded59-f2c7-4e1d-863b-c790e7eab21a

5. **Generating Streaming Data**: We need to generate streaming data that can be published to Pub Sub. Then those messages will be picked to be processed by the pipeline. To generate data we will use **random()** library to create input messages. Using the generating_data.py we will be able to generate random data in the required format. This generated data will be published to Pub/Sub using publish_to_pubsub.py. Here we will use PublisherClient object, add the path to the topic using the topic_path method and call the publish_to_pubsub() function while passing the topic_path and data.

```python
import random

LINE ="""   {Existing_account} 
        {Duration_month} 
        {Credit_history} 
        {Purpose} 
        {Credit_amount} 
        .....
        {Foreign_worker}"""

def generate_log():
existing_account = ['B11','A12','C14',
                    'D11','E11','A14',
                    'G12','F12','A11',
                    'H11','I11',
                    'J14','K14','L11',
                    'A13'
                   ]
Existing_account = random.choice(existing_account)

duration_month = []
for i  in range(6, 90 , 3):
    duration_month.append(i)
Duration_month = random.choice(duration_month)
....
Foreign_worker = ['A201',
                'A202']
Foreign_worker = random.choice(foreign_worker)
log_line = LINE.format(
    Existing_account=Existing_account,
    Duration_month=Duration_month,
    Credit_history=Credit_history,
    Purpose=Purpose,
    ...
    Foreign_worker=Foreign_worker
)

return log_line

```

##### Related code
1. [generating_data.py](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/generating_data.py)

## Pipeline construction

### 1. **Reading the Data**

Now we will go step by step to create a pipeline starting with reading the data. The data is read using **beam.io.ReadFromText()**. Here we will just read the input values and save it in a file. The output is stored in text file named simpleoutput.

```python
def run(argv=None, save_main_session=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--input',
      dest='input',
      help='Input file to process')
    parser.add_argument(
      '--output',
      dest='output',
      default='../output/result.txt',
      help='Output file to write results to.')
    known_args, pipeline_args = parser.parse_known_args(argv)
    options = PipelineOptions(pipeline_args)
    with beam.Pipeline(options=PipelineOptions()) as p:
        data = (p 
                | 'Read Data' >> beam.io.ReadFromText(known_args.input)
                | 'Filter Header' >> beam.Filter(lambda line: not line.startswith("Existing account"))
            ) 
if __name__ == '__main__':
    run()
``` 

### 2. **Create Batch Dataflow Job**

Now the we will be constructing the Dataflow job what will pull the data from GCS bucket and injest into Bigquery. The code for is it persent [here](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/batch-pipeline.py)
   
- ***Parsing the data***: After reading the input file we will split the data using split(). Data is segregated into different columns to be used in further steps. We will **ParDo()** to create a split function. The output of this step is present in SplitPardo text file.

    ```python
        class Split(beam.DoFn):
            #This Function Splits the Dataset into a dictionary
            def process(self, element): 
                Existing_account,
                Duration_month,
                Credit_history,
                Purpose,
                Credit_amount,
                Saving,
                Employment_duration,
                Installment_rate,
                Personal_status,
                Debtors,
                Residential_Duration,
                Property,
                Age,
                Installment_plans,
                Housing,
                Number_of_credits
                Job,
                Liable_People,
                Telephone,
                Foreign_worker,
                Classification = element.split(' ')
             return [{
                'Existing_account': str(Existing_account),
                'Duration_month': int(Duration_month),
                'Credit_history': str(Credit_history),
                'Purpose': str(Purpose),
                'Credit_amount': int(Credit_amount),
                'Saving': str(Saving),
                'Employment_duration':str(Employment_duration),
                'Installment_rate': int(Installment_rate),
                'Personal_status': str(Personal_status),
                'Debtors': str(Debtors),
                'Residential_Duration': int(Residential_Duration),
                'Property': str(Property),
                'Age': int(Age),
                'Installment_plans':str(Installment_plans),
                'Housing': str(Housing),
                'Number_of_credits': int(Number_of_credits),
                'Job': str(Job),
                'Liable_People': int(Liable_People),
                'Telephone': str(Telephone),
                'Foreign_worker': str(Foreign_worker),
                'Classification': int(Classification)
            }]
        def run(argv=None, save_main_session=True):
            ...
            with beam.Pipeline(options=PipelineOptions()) as p:
                data = (p 
                         | beam.io.ReadFromText(known_args.input) )
                parsed_data = (data 
                         | 'Parsing Data' >> beam.ParDo(Split())
                         | 'Writing output' >> beam.io.WriteToText(known_args.output))
    
        if __name__ == '__main__':
            run()
    ``` 

- ***Filtering the data***: Now we will clean the data by removing all the rows having Null values from the dataset. We will use **Filter()** to return only valid rows with no Null values. Output of this step is saved in the file named Filtered_data.

    ```python
        ...
        def Filter_Data(data):
        #This will remove rows the with Null values in any one of the columns
            return data['Purpose'] !=  'NULL' 
            and len(data['Purpose']) <= 3  
            and data['Classification'] !=  'NULL' 
            and data['Property'] !=  'NULL' 
            and data['Personal_status'] != 'NULL' 
            and data['Existing_account'] != 'NULL' 
            and data['Credit_amount'] != 'NULL' 
            and data['Installment_plans'] != 'NULL'
        ...
        def run(argv=None, save_main_session=True):
            ...
            with beam.Pipeline(options=PipelineOptions()) as p:
                data = (p 
                         | beam.io.ReadFromText(known_args.input) )
                parsed_data = (data 
                         | 'Parsing Data' >> beam.ParDo(Split()))
                filtered_data = (parsed_data
                         | 'Filtering Data' >> beam.Filter(Filter_Data)          
                         | 'Writing output' >> beam.io.WriteToText(known_args.output))
    
        if __name__ == '__main__':
            run()
    ```

- ***Performing Type Convertion***: After Filtering we will convert the datatype of numeric columns from String to Int or Float datatype. Here we will use **Map()** to apply the Convert_Datatype(). The output of this step is saved in Convert_datatype text file.

    ```python
        ... 
        def Convert_Datatype(data):
            #This will convert the datatype of columns from String to integers or Float values
            data['Duration_month'] = int(data['Duration_month']) if 'Duration_month' in data else None
            data['Credit_amount'] = float(data['Credit_amount']) if 'Credit_amount' in data else None
            data['Installment_rate'] = int(data['Installment_rate']) if 'Installment_rate' in data else None
            data['Residential_Duration'] = int(data['Residential_Duration']) if 'Residential_Duration' in data else None
            data['Age'] = int(data['Age']) if 'Age' in data else None
            data['Number_of_credits'] = int(data['Number_of_credits']) if 'Number_of_credits' in data else None
            data['Liable_People'] = int(data['Liable_People']) if 'Liable_People' in data else None
            data['Classification'] =  int(data['Classification']) if 'Classification' in data else None
           
            return data
        ...
        def run(argv=None, save_main_session=True):
            ...
            with beam.Pipeline(options=PipelineOptions()) as p:
                data = (p 
                         | beam.io.ReadFromText(known_args.input) )
                parsed_data = (data 
                         | 'Parsing Data' >> beam.ParDo(Split()))
                filtered_data = (parsed_data
                         | 'Filtering Data' >> beam.Filter(Filter_Data))
                Converted_data = (filtered_data
                         | 'Convert Datatypes' >> beam.Map(Convert_Datatype)
                         | 'Writing output' >> beam.io.WriteToText(known_args.output))
    
        if __name__ == '__main__':
            run()
    ```

### 3. **Inserting Data in Bigquery**

Final step in the Pipeline it to insert the data in Bigquery. To do this we will use **beam.io.WriteToBigQuery()** which requires Project id and a Schema of the target table to save the data. 

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import argparse

SCHEMA = '
        Existing_account:STRING,
        Duration_month:INTEGER,
        Credit_history:STRING,
        Purpose:STRING,
        Credit_amount:FLOAT,
        Saving:STRING,
        Employment_duration:STRING,
        Installment_rate:INTEGER,
        Personal_status:STRING,
        Debtors:STRING,
        Residential_Duration:INTEGER,
        Property:STRING,
        Age:INTEGER,
        Installment_plans:STRING,
        Housing:STRING,
        Number_of_credits:INTEGER,
        Job:STRING,
        Liable_People:INTEGER,
        Telephone:STRING,
        Foreign_worker:STRING,
        Classification:INTEGER
        '
...
def run(argv=None, save_main_session=True):
    ...
    parser.add_argument(
      '--project',
      dest='project',
      help='Project used for this Pipeline')
    ...
    PROJECT_ID = known_args.project
    with beam.Pipeline(options=PipelineOptions()) as p:
        data = (p 
                | 'Read Data' >> beam.io.ReadFromText(known_args.input)
                | 'Filter Header' >> beam.Filter(lambda line: not line.startswith("Existing account"))
            )
    parsed_data = (data 
                 | 'Parsing Data' >> beam.ParDo(Split()))
    filtered_data = (parsed_data
                 | 'Filtering Data' >> beam.Filter(Filter_Data))
    Cleaned_data = (filtered_data
                 | 'Convert Datatypes' >> beam.Map(Convert_Datatype))
    output =( Cleaned_data      
                 | 'Writing to bigquery' >> beam.io.WriteToBigQuery(
                   '{0}:GermanCredit.GermanCreditTable'.format(PROJECT_ID),
                   schema=SCHEMA,
                   write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)

if __name__ == '__main__':
    run()        
```
##### Related code
1. [batch-pipeline.py](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/batch-pipeline.py)

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/fadb5172-8c24-40f9-aee0-190a2562d170


### 4. **Create Kubeflow Pipeline**

Now the real pipeline creating starts. Here will we will try to create pipeline components one by one. File to be used is training_pipeline.ipynb

- ***Ingest Data*** : First step in the pipeline is Data ingestion. Here we simply read the data german_data.csv from our bucket. This method expect 2 arguments, one is input_data_path coming from input arguments of the python job and second is the output_dataset path to copy file output path

    ```python
        import yaml
        from kfp import dsl
        from kfp.dsl import (
            component,
            Metrics,
            Dataset,
            Input,
            Model,
            Artifact,
            OutputPath,
            Output,
        )
        from kfp import compiler
        import google.cloud.aiplatform as aiplatform
        import os
        @component(
            base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
            )
        def data_ingestion(input_data_path: str,
                           project_id: str,
                           region: str, 
                           input_data: Output[Dataset],):
            import pandas as pd
            from datetime import datetime, timedelta
            from google.cloud import bigquery
            import logging
            client = bigquery.Client(project=project_id, location=region)
            sql = """
            SELECT *
            FROM `{}.GermanCredit.GermanCreditTable`
            """.format(project_id)
            df = client.query_and_wait(sql).to_dataframe()
            df.to_csv(input_data.path, index=False)
    ```
    
- ***Preprocess Data***: Second step in the pipeline is preprocess the data. Here we simply clean the data provided in output of previous step. This method expect 2 arguments, one is training_df coming from previous output of the python job and second is the output_dataset path to copy preprocessed file. Here we first convert reduce skewness in the data using logarithmic transform. Then we perform MinMAxscaling. At the end we perform label encoding to categorical columns. Then we write output data to output path.  

    ```python  
        @component(
            base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
        )
        
        def preprocessing(train_df: Input[Dataset],
                  gcs_bucket: str,
                  input_data_preprocessed: Output[Dataset]):
            import pandas as pd
            import numpy as np
            import os
            import joblib
            from sklearn.preprocessing import MinMaxScaler, LabelEncoder
            from google.cloud import storage
        
            # --- CONFIG ---
            GCS_ENCODER_FOLDER = "label_encoders"
            LOCAL_ENCODER_DIR = "/encoders"
            os.makedirs(LOCAL_ENCODER_DIR, exist_ok=True)
        
            def upload_encoder_to_gcs(local_path, gcs_path):
                client = storage.Client()
                bucket = client.bucket(gcs_bucket)
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to gs://{gcs_bucket}/{gcs_path}")
        
            def encode_columns(df, columns):
                encoders = {}
                for column in columns:
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])
                    encoders[column] = le
        
                    # Save encoder
                    local_file = f"{LOCAL_ENCODER_DIR}/{column}_label_encoder.pkl"
                    joblib.dump(le, local_file)
        
                    # Upload to GCS
                    upload_encoder_to_gcs(
                        local_file,
                        f"{GCS_ENCODER_FOLDER}/{column}_label_encoder.pkl"
                    )
                return df
        
            def preprocess(df):
                numeric_columns = df.describe().columns
                df_log_transformed = df.copy()
                df_log_transformed[numeric_columns] = df[numeric_columns].apply(lambda x: np.log(x + 1))
                scaler = MinMaxScaler()
                df_scaled_log_transformed = df_log_transformed.copy()
                df_scaled_log_transformed[numeric_columns] = scaler.fit_transform(df_scaled_log_transformed[numeric_columns])
                categorical_columns = [
                    'Existing_account', 'Credit_history', 'Purpose', 'Saving',
                    'Employment_duration', 'Personal_status', 'Debtors', 'Property',
                    'Installment_plans', 'Housing', 'Job', 'Telephone', 'Foreign_worker'
                ]
                df_scaled_log_transformed = encode_columns(df_scaled_log_transformed, categorical_columns)
                return df_scaled_log_transformed
        
            # Load, preprocess, save output
            df = pd.read_csv(train_df.path)
            df = preprocess(df)
            df.to_csv(input_data_preprocessed.path, index=False)
    ```
    
- ***Split Data into Training and Testing Dataset***: Third step in the pipeline is split data into training and test datasets. We split the datasets train and test and save them to output paths. This methods has 5 arguments, one for dataset input from previous steps, label, output train and test dataset paths and test size. 
      
    ```python
        @component(
            base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
        )
        def train_test_data_split(
            dataset_in: Input[Dataset],
            target_column: str,
            dataset_train: Output[Dataset],
            dataset_test: Output[Dataset],
            test_size: float = 0.2,
        ):
            import pandas as pd
            import logging
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            import pandas as pd
            from sklearn.utils import shuffle
            def get_train_test_splits(df, target_column, test_size_sample ):
                df = shuffle(df)
                x = df.drop(target_column, axis=1)
                y = df[target_column]
        
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_sample)
        
                X_train = pd.DataFrame(X_train)
                y_train = pd.DataFrame(y_train)
                X_test = pd.DataFrame(X_test)
                y_test = pd.DataFrame(y_test)
                X_train.reset_index(drop=True, inplace=True)
                y_train.reset_index(drop=True, inplace=True)
                X_test = X_test.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)
                X_train = pd.concat([X_train, y_train], axis=1)
                X_test = pd.concat([X_test, y_test], axis=1)
                X_train.columns = x.columns.to_list() + [target_column]
                X_test.columns = x.columns.to_list() + [target_column]
                return X_train, X_test
            data = pd.read_csv(dataset_in.path)
            X_train, X_test = get_train_test_splits(
                data, target_column, test_size
            )
            X_train.to_csv(dataset_train.path, index=False)
            X_test.to_csv(dataset_test.path, index=False)
    ```
    
- ***HyperParametering Tuning***: Fourth step in the pipeline is perform hyperparameter tuning. Here we try to find the optimal settings for its parameters to improve model performance. We try to find best parameters for Logistic regression. 

    ```python
        @component(
             base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
        )
        def hyperparameters_training(
            dataset_train: Input[Dataset],
            dataset_test: Input[Dataset],
            target: str,
            max_evals: int,
            metrics: Output[Metrics],
            param_artifact: Output[Artifact],
            ml_model: Output[Model],
        ):
            import pandas as pd
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
            import joblib
            import os
            import json
            import logging
            
        
            X_train = pd.read_csv(dataset_train.path)
            X_test = pd.read_csv(dataset_test.path)
        
            y_train = X_train[target]
            y_test = X_test[target]
            X_train = X_train.drop(target, axis=1)
            X_test = X_test.drop(target, axis=1)
            space = {
                'C': hp.loguniform('C', -3, 3),  # log-uniform between ~0.05 to ~20
                'penalty': hp.choice('penalty', ['l1', 'l2']),  # safer to exclude 'elasticnet' unless solver == 'saga'
                'solver': hp.choice('solver', ['liblinear', 'saga']),  # only solvers that support l1
                'class_weight': hp.choice('class_weight', [None, 'balanced']),
                'max_iter': hp.choice('max_iter', [100, 1000,2500, 5000]),
            }
            def objective(params):
                rf = LogisticRegression(**params)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
        
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
        
                metrics.log_metric("accuracy", accuracy)
                metrics.log_metric("precision", precision)
                metrics.log_metric("recall", recall)
                metrics.log_metric("f1", f1)
        
                return {'loss': -accuracy, 'status': STATUS_OK, 'model': rf}
            trials = Trials()
            
            best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        
            best_params = trials.best_trial['result']['model'].get_params()
            best_model = trials.best_trial['result']['model']
        
            # Save the best model
            os.makedirs(ml_model.path, exist_ok=True)
            joblib.dump(best_model, os.path.join(ml_model.path, 'model.joblib'))
        
            # Save the best hyperparameters
            with open(param_artifact.path, "w") as f:
                json.dump(best_params, f)
    ```
    
- ***Deploy Model to Model Registry***: Fifth step in the pipeline is deploy the model to Model Registry. This will help us use the model for online prediction. Here we either deploy latest version of the model or a new model depending if the model is already existing. Here we have 6 arguments in the method, starting with project, region coming from input arguments, ml_model created in previous step, model_name to be deployed in, serving container image to be used to deploy the code, and model_uri to be used to create the Endpoint in next step.

    ```python
        @component(
            base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
        )
        def deploy_model(
            project: str,
            region: str,
            ml_model: Input[Model],
            model_name: str,
            serving_container_image_uri: str,
            model_uri: Output[Artifact],
        ):
            from google.cloud import aiplatform
            import logging
            import os
        
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logger = logging.getLogger(__name__)
        
            existing_models = aiplatform.Model.list(
                filter=f"display_name={model_name}", project=project, location=region
            )
            if existing_models:
                latest_model = existing_models[0]
                logger.info(f"Creating a new version for existing model: {latest_model.name}")
                model = aiplatform.Model.upload(
                    display_name=model_name,
                    artifact_uri=ml_model.path,
                    location='asia-south1',
                    serving_container_image_uri=serving_container_image_uri,
                    parent_model=latest_model.resource_name,
                )
            else:
                logger.info("No existing model found. Creating a new model.")
                model = aiplatform.Model.upload(
                    display_name=model_name,
                    artifact_uri=ml_model.path,
                    location='asia-south1',
                    serving_container_image_uri=serving_container_image_uri,
                )
            os.makedirs(model_uri.path, exist_ok=True)
            with open(os.path.join(model_uri.path, "model_uri.txt"), "w") as f:
                f.write(model.resource_name)
    ```
    
- ***Create Endpoint for Online Prediction***: In the sixth step we simply create an endpoint to be used for online prediction. This method has project, region, model_name and model_uri as input arguments.

    ```python
        @component(
            base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
        )
        def create_endpoint(
            project: str,
            region: str,
            model_name: str,
            model_uri: Input[Artifact],
        ):
            from google.cloud import aiplatform
            import logging
            import os
            with open(os.path.join(model_uri.path, "model_uri.txt"), "r") as f:
                model_resource_name = f.read()
            model = aiplatform.Model(model_resource_name)
            traffic_split = {"0": 100}
            machine_type = "n1-standard-4"
            min_replica_count = 1
            max_replica_count = 1
            
            endpoint = model.deploy(
                    deployed_model_display_name=model_name,
                    machine_type=machine_type,
                    traffic_split = traffic_split,
                    min_replica_count=min_replica_count,
                    max_replica_count=max_replica_count
                )
        
    ```
    
- ***Defining the pipeline***: At last we define the pipeline components.
      
    ```python
        @dsl.pipeline(name="Training Pipeline", pipeline_root="gs://demo_bucket_kfl/pipeline_root_demo")
        def pipeline(
            input_data_path: str = "gs://demo_bucket_kfl/german_data.csv",
            project_id: str = "solar-dialect-264808",
            region: str = "asia-south1",
            model_name: str = "demo_model",
            target: str = "Classification",
            gcs_bucket: str = "demo_bucket_kfl",
            max_evals: int = 30,
            use_hyperparameter_tuning: bool = True,
            serving_container_image_uri: str = "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
        ):
            data_op = data_ingestion(
                input_data_path=input_data_path)
            data_op.set_caching_options(False)
        
            data_preprocess_op = preprocessing(train_df=data_op.outputs["input_data"], gcs_bucket=gcs_bucket)
            data_preprocess_op.set_caching_options(False)
            train_test_split_op = train_test_data_split(
                dataset_in=data_preprocess_op.outputs["input_data_preprocessed"],
                target_column="Classification",
                test_size=0.2,
            )
            train_test_split_op.set_caching_options(False)
            hyperparam_tuning_op = hyperparameters_training(
                dataset_train=train_test_split_op.outputs["dataset_train"],
                dataset_test=train_test_split_op.outputs["dataset_test"],
                target=target,
                max_evals=max_evals
            )
            hyperparam_tuning_op.set_caching_options(False)
            deploy_model_op = deploy_model(
                project=project_id, region=region,
                ml_model=hyperparam_tuning_op.outputs["ml_model"],
                model_name=model_name,
                serving_container_image_uri=serving_container_image_uri
            )
            deploy_model_op.set_caching_options(False)
            create_endpoint_op = create_endpoint(
                project=project_id, region=region,
                model_name=model_name,
                model_uri = deploy_model_op.outputs["model_uri"]
            )
            create_endpoint_op.set_caching_options(False)
            
        if __name__ == "__main__":
            compiler.Compiler().compile(pipeline_func=pipeline, package_path="training_pipeline.json")
    ```
##### Related Code
1. [training_pipeline.py](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/training_pipeline.py)

    
- ***Running the Pipeline***: Now we simply have to run the pipeline. 

    ```python
        from google.cloud.aiplatform import PipelineJob
        
        pipeline_job = PipelineJob(
            display_name="training_pipeline_job",
            template_path="training_pipeline.json",
            pipeline_root="gs://demo_bucket_kfl/pipeline_root_demo",
            project="solar-dialect-264808",
            location="asia-south1",
            )
        pipeline_job.run()
    ```
##### Related code
1. [run_pipeline.ipynb](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/run_pipeline.ipynb)

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/fb8a7cd2-2bd5-4127-8540-391058a45f8e


- ***Metadata Management***: At last we simply verify the metadata of the pipeline. The output artifacts are provided in output artifacts folder.

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/944b2b5d-cf57-4817-bfa7-87f4496b55d6


### 5. **Reading Data from Pub Sub**

Now we will start reading data from Pub sub to start the pipeline. The data is read using **beam.io.ReadFromPubSub()**. Here we will just read the input message by providing the TOPIC and the output is decoded which was encoded while generating the data. 

```python
def run(argv=None, save_main_session=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project',
        dest='project',
        help='Project used for this Pipeline')
    parser.add_argument(
        '--bucket_name', 
        required=True, 
        help='The name of the bucket')
    parser.add_argument(
        '--input_subscription',
        help=('Input PubSub subscription of the form '
              '"projects/<PROJECT>/subscriptions/<SUBSCRIPTION>."'))
    parser.add_argument(
        '--input_topic',
        help=('Input PubSub topic of the form '
              '"projects/<PROJECT>/topics/<TOPIC>".'))
    known_args, pipeline_args = parser.parse_known_args(argv)
    options = PipelineOptions(pipeline_args)
    PROJECT_ID = known_args.project
    TOPIC = known_args.input_topic
    SUBSCRIPTION = known_args.input_subscription
    with beam.Pipeline(options=PipelineOptions()) as p:
        Encoded_data   = (p 
                       | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes) )
        Data           = ( Encoded_data
                      | 'Decode' >> beam.Map(lambda x: x.decode('utf-8') ) )
                               ) 
        if __name__ == '__main__':
            run()
```
    
### 6. **Create Streaming Dataflow Job**

Now the we will be constructing the Dataflow job what will pull the data from Pub/sub , perform ETL and pull inference from Vertex AI endpoint and injest into Bigquery. The code for is it persent [here](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/ml-streaming-pipeline-endpoint.py)

- **Parsing the data**: After reading the input from Pub-Sub we will split the data using split(). Data is segregated into different columns to be used in further steps. We will **ParDo()** to create a split function.

    ```python
        class Split(beam.DoFn):
            #This Function Splits the Dataset into a dictionary
            def process(self, element): 
                Existing_account,
                Duration_month,
                Credit_history,
                Purpose,
                Credit_amount,
                Saving,
                Employment_duration,
                Installment_rate,
                Personal_status,
                Debtors,
                Residential_Duration,
                Property,
                Age,
                Installment_plans,
                Housing,
                Number_of_credits
                Job,
                Liable_People,
                Telephone,
                Foreign_worker= element.split(' ')
            return [{
               'Existing_account': str(Existing_account),
                'Duration_month': int(Duration_month),
                'Credit_history': str(Credit_history),
                'Purpose': str(Purpose),
                'Credit_amount': int(Credit_amount),
                'Saving': str(Saving),
                'Employment_duration':str(Employment_duration),
                'Installment_rate': int(Installment_rate),
                'Personal_status': str(Personal_status),
                'Debtors': str(Debtors),
                'Residential_Duration': int(Residential_Duration),
                'Property': str(Property),
                'Age': int(Age),
                'Installment_plans':str(Installment_plans),
                'Housing': str(Housing),
                'Number_of_credits': int(Number_of_credits),
                'Job': str(Job),
                'Liable_People': int(Liable_People),
                'Telephone': str(Telephone),
                'Foreign_worker': str(Foreign_worker)
            }]
        def run(argv=None, save_main_session=True):
            ...
            with beam.Pipeline(options=PipelineOptions()) as p:
                Encoded_data   = (p 
                       | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes) )
                Data           = ( Encoded_data
                              | 'Decode' >> beam.Map(lambda x: x.decode('utf-8') ) )
                Parsed_data    = (Data 
                               | 'Parsing Data' >> beam.ParDo(Split()))
                                | 'Writing output' >> beam.io.WriteToText(known_args.output)
                               )
    
        if __name__ == '__main__':
            run()
    ```
- **Filtering Data**: After dattype coversion we will filter the unrequired data .
     
    ```python
        def Filter_Data(data):
            return data['Purpose'] !=  'NULL' and 
                    len(data['Purpose']) <= 3  and  
                    data['Property'] !=  'NULL' and 
                    data['Personal_status'] != 'NULL' and 
                    data['Existing_account'] != 'NULL' and 
                    data['Credit_amount'] != 'NULL' and 
                    data['Installment_plans'] != 'NULL'
        ...
        def run(argv=None, save_main_session=True):
            ...
            with beam.Pipeline(options=PipelineOptions()) as p:
                 encoded_data = ( p 
                                | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes) 
                                )
                        data =  ( encoded_data
                                | 'Decode' >> beam.Map(lambda x: x.decode('utf-8') 
                                ) 
                  parsed_data = ( data 
                                | 'Parsing Data' >> beam.ParDo(Split())
                                )
                   filtered_data = (Parsed_data
                                 | 'Filtering Data' >> beam.Filter(Filter_Data))
                                | 'Writing output' >> beam.io.WriteToText(known_args.output))
    
        if __name__ == '__main__':
            run()
    ```
    
- **Lable Encoding Data**: After datatype coversion we will label encode the data to make it machine learning worthy
     
    ```python
        class ApplyLabelEncoding(beam.DoFn):
            def __init__(self, bucket_name, encoder_folder, columns_to_encode):
                self.bucket_name = bucket_name
                self.encoder_folder = encoder_folder
                self.columns_to_encode = columns_to_encode
                self.encoders = {}
        
            def setup(self):
                # Load encoders only once per worker
                from google.cloud import storage
                import tempfile
        
                client = storage.Client()
                bucket = client.bucket(self.bucket_name)
        
                for column in self.columns_to_encode:
                    blob_path = f"{self.encoder_folder}/{column}_label_encoder.pkl"
                    blob = bucket.blob(blob_path)
        
                    with tempfile.NamedTemporaryFile() as f:
                        blob.download_to_filename(f.name)
                        self.encoders[column] = joblib.load(f.name)
        
            def process(self, element):
                for col in self.columns_to_encode:
                    if col in element and element[col] in self.encoders[col].classes_:
                        element[col] = int(self.encoders[col].transform([element[col]])[0])
                    else:
                        element[col] = -1  # Unknown or unseen value
                yield element
        ...
        def run(argv=None, save_main_session=True):
            ...
            with beam.Pipeline(options=PipelineOptions()) as p:
                 encoded_data = ( p 
                                | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes) 
                                )
                        data =  ( encoded_data
                                | 'Decode' >> beam.Map(lambda x: x.decode('utf-8') 
                                ) 
                    parsed_data = ( data 
                                | 'Parsing Data' >> beam.ParDo(Split())
                                )
                    filtered_data = (Parsed_data
                                 | 'Filtering Data' >> beam.Filter(Filter_Data)))
                    Encoded_data = (filtered_data 
                                | 'Label Encoding' >> beam.ParDo(ApplyLabelEncoding(
                                    bucket_name=known_args.bucket_name,
                                    encoder_folder="label_encoders",
                                    columns_to_encode=categorical_columns
                                ))
                                | 'Writing output' >> beam.io.WriteToText(known_args.output))
    
        if __name__ == '__main__':
            run()
    ```

- **Performing Type Convertion**: After parsing we will convert the datatype of numeric columns from String to Int or Float datatype. Here we will use **Map()** to apply the Convert_Datatype(). 

    ```python
        ... 
        def Convert_Datatype(data):
            #This will convert the datatype of columns from String to integers or Float values
            data['Duration_month'] = int(data['Duration_month']) if 'Duration_month' in data else None
            data['Credit_amount'] = float(data['Credit_amount']) if 'Credit_amount' in data else None
            data['Installment_rate'] = int(data['Installment_rate']) if 'Installment_rate' in data else None
            data['Residential_Duration'] = int(data['Residential_Duration']) if 'Residential_Duration' in data else None
            data['Age'] = int(data['Age']) if 'Age' in data else None
            data['Number_of_credits'] = int(data['Number_of_credits']) if 'Number_of_credits' in data else None
            data['Liable_People'] = int(data['Liable_People']) if 'Liable_People' in data else None
            return data
        ...
        def run(argv=None, save_main_session=True):
            ...
            with beam.Pipeline(options=PipelineOptions()) as p:
                 encoded_data = ( p 
                                | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes) 
                                )
                        data =  ( encoded_data
                                | 'Decode' >> beam.Map(lambda x: x.decode('utf-8') 
                                ) 
                   parsed_data = ( data 
                                | 'Parsing Data' >> beam.ParDo(Split())
                                )
                   filtered_data = (Parsed_data
                                 | 'Filtering Data' >> beam.Filter(Filter_Data)))
                    Encoded_data = (filtered_data 
                                | 'Label Encoding' >> beam.ParDo(ApplyLabelEncoding(
                                    bucket_name=known_args.bucket_name,
                                    encoder_folder="label_encoders",
                                    columns_to_encode=categorical_columns
                                ))
                   Converted_data = ( Encoded_data
                                    | 'Convert Datatypes' >> beam.Map(Convert_Datatype)
                                    | 'Writing output' >> beam.io.WriteToText(known_args.output))
    
        if __name__ == '__main__':
            run()
    ```
    
### 7. **Online Prediction using Vertex AI Endpoint**

Now we will perform predictions from the machine learning model. If you wish to learn how this machine learning model was created, please visit this [repository](https://github.com/adityasolanki205/German-Credit). We will predict customer segment using endpoint created in Kubeflow.


```python
... 
def call_vertex_ai(data):
     aiplatform.init(project='827249641444', location='asia-south1')
     endpoints = aiplatform.Endpoint.list()
     # Display endpoint info
     for endpoint in endpoints:
         print(f"Resource Name: {endpoint.resource_name}")
         endpoint_created = endpoint.resource_name
   
     feature_order = ['Existing_account', 'Duration_month', 'Credit_history', 'Purpose',
                  'Credit_amount', 'Saving', 'Employment_duration', 'Installment_rate',
                  'Personal_status', 'Debtors', 'Residential_Duration', 'Property', 'Age',
                  'Installment_plans', 'Housing', 'Number_of_credits', 'Job', 
                  'Liable_People', 'Telephone', 'Foreign_worker']
     endpoint = aiplatform.Endpoint(endpoint_name=endpoint_created)
     features = [data[feature] for feature in feature_order]
     response = endpoint.predict(
         instances=[features]
     )
   
     prediction = response.predictions[0]
     data['Classification'] = int(prediction)
     return data
...
def run(argv=None, save_main_session=True):
    ...
    with beam.Pipeline(options=PipelineOptions()) as p:
       Encoded_data   = (p 
               | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes) )
        Data           = ( Encoded_data
                      | 'Decode' >> beam.Map(lambda x: x.decode('utf-8') ) )
        Parsed_data    = (Data 
                       | 'Parsing Data' >> beam.ParDo(Split()))
        filtered_data = (Parsed_data
                     | 'Filtering Data' >> beam.Filter(Filter_Data))
        Encoded_data = (filtered_data 
                    | 'Label Encoding' >> beam.ParDo(ApplyLabelEncoding(
                        bucket_name=known_args.bucket_name,
                        encoder_folder="label_encoders",
                        columns_to_encode=categorical_columns
                    ))
                )
        Converted_data = (Encoded_data
                       | 'Convert Datatypes' >> beam.Map(Convert_Datatype))
        Prediction   = (Converted_data
                        |'Get Inference' >> beam.Map(call_vertex_ai))
                         | 'Saving the output' >> beam.io.WriteToText(known_args.output))
if __name__ == '__main__':
    run()
```

### 8. **Inserting Data in Bigquery**

Final step in the Pipeline it to insert the data in Bigquery. To do this we will use **beam.io.WriteToBigQuery()** which requires Project id and a Schema of the target table to save the data. 

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import argparse

SCHEMA =
'
Existing_account:STRING,
Duration_month:INTEGER,
Credit_history:STRING,
Purpose:STRING,
Credit_amount:FLOAT,
Saving:STRING,
Employment_duration:STRING,
Installment_rate:INTEGER,
Personal_status:STRING,
Debtors:STRING,
Residential_Duration:INTEGER,
Property:STRING,
Age:INTEGER,
Installment_plans:STRING,
Housing:STRING,
Number_of_credits:INTEGER,
Job:STRING,
Liable_People:INTEGER,
Telephone:STRING,
Foreign_worker:STRING,
Classification:INTEGER
'
...
def run(argv=None, save_main_session=True):
    ...
    parser.add_argument(
      '--project',
      dest='project',
      help='Project used for this Pipeline')
    ...
    PROJECT_ID = known_args.project
    with beam.Pipeline(options=PipelineOptions()) as p:
        Encoded_data   = (p 
                   | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes) )
        Data           = ( Encoded_data
                      | 'Decode' >> beam.Map(lambda x: x.decode('utf-8') ) )
        Parsed_data    = (Data 
                       | 'Parsing Data' >> beam.ParDo(Split()))
        filtered_data = (Parsed_data
                     | 'Filtering Data' >> beam.Filter(Filter_Data))
        Encoded_data = (filtered_data 
                    | 'Label Encoding' >> beam.ParDo(ApplyLabelEncoding(
                        bucket_name=known_args.bucket_name,
                        encoder_folder="label_encoders",
                        columns_to_encode=categorical_columns
                    ))
                )
        Converted_data = (Encoded_data
                       | 'Convert Datatypes' >> beam.Map(Convert_Datatype))
        Prediction   = (Converted_data
                        |'Get Inference' >> beam.Map(call_vertex_ai))
        output         = ( Prediction      
                       | 'Writing to bigquery' >> beam.io.WriteToBigQuery(
                       '{0}:GermanCredit.GermanCreditTable'.format(PROJECT_ID),
                       schema=SCHEMA,
                       write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
                      )

if __name__ == '__main__':
    run()        
```

##### Related Code
1. [ml-streaming-pipeline-endpoint.py](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/ml-streaming-pipeline-endpoint.py)
2. [update_python.ipynb](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/update_python.ipynb)

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/949eb305-1f9f-44bc-b010-7b9811e9c51f

### 9. **Model Monitoring using Vertex AI**

To monitor prediction quality and detect drift in real-time, we use **Vertex AI Model Monitoring** on the deployed endpoint. This helps detect output drift or performance degradation based on predictions stored in **BigQuery**. The monitoring is configured to observe distribution changes over time. Schema is as follows:

```yaml
{
"featureFields": [
{
  "name": "Existing_account",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Duration_month",
  "dataType": "integer",
  "repeated": false
},
{
  "name": "Credit_history",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Purpose",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Credit_amount",
  "dataType": "float",
  "repeated": false
},
{
  "name": "Saving",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Employment_duration",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Installment_rate",
  "dataType": "integer",
  "repeated": false
},
{
  "name": "Personal_status",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Debtors",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Residential_Duration",
  "dataType": "integer",
  "repeated": false
},
{
  "name": "Property",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Age",
  "dataType": "integer",
  "repeated": false
},
{
  "name": "Installment_plans",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Housing",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Number_of_credits",
  "dataType": "integer",
  "repeated": false
},
{
  "name": "Job",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Liable_People",
  "dataType": "integer",
  "repeated": false
},
{
  "name": "Telephone",
  "dataType": "string",
  "repeated": false
},
{
  "name": "Foreign_worker",
  "dataType": "string",
  "repeated": false
}
],
"predictionFields": [
{
  "name": "Classification",
  "dataType": "integer",
  "repeated": false
}
],
"groundTruthFields": []
}
```
##### Related code
1. [model_monitoring_schema.txt](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/model_monitoring_schema.txt)

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/b5ca4e5a-d826-4089-a6b4-ddb0e43a1913

### 10. **Triggering Cloud Alerts on Threshold Breach**

Vertex AI Model Monitoring is connected to **Cloud Monitoring**, which is set up with custom alert policies. When thresholds for prediction drift are breached, **Cloud Alerting** triggers an incident, sending a message to a **Pub/Sub** topic for further automated action like retraining.

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/9910e724-92e2-4950-86a0-0e2bd6de07e1

### 11. **Handling Alerts using Cloud Run Functions**

The **Pub/Sub** message is consumed by a **Cloud Run Function**, which initiates the retraining pipeline using **Vertex AI Pipelines**. The retraining is triggered based on the message received from the alert system.
   
```python
import base64
import json
from google.cloud import aiplatform
from google.cloud.aiplatform import PipelineJob

PROJECT_ID = ""                     # <---CHANGE THIS
REGION = "asia-south1"                            # <---CHANGE THIS
PIPELINE_ROOT ="" # <---CHANGE THIS

def subscribe(event, context):
    # decode the event payload string
    payload_message = base64.b64decode(event['data']).decode('utf-8')
    # parse payload string into JSON object
    payload_json = json.loads(payload_message)
    # trigger pipeline run with payload
    trigger_retraining(payload_json)

def trigger_retraining(payload_json):
    pipeline_spec_uri = "gs://<bucket>/training_pipeline.json"
    
    # Create a PipelineJob using the compiled pipeline from pipeline_spec_uri
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
    )
    
    pipeline_job = PipelineJob(
        display_name="retraining_pipeline_job",
        template_path=pipeline_spec_uri,
        pipeline_root=PIPELINE_ROOT
        )
    pipeline_job.run()
```
##### Related Code
1. [trigger_retraining.py](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/trigger_retraining.py)
2. [cloud_run_requirement.txt](https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline/blob/main/cloud_run_requirement.txt)

### 12. **Automated Model Retraining via Cloud Run Functions**

In this final step, **Cloud Run Functions** initiate a new training job by triggering the same or a different pipeline. This step closes the feedback loop, enabling **end-to-end automation of model monitoring and retraining** to maintain model accuracy over time.

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/e7e26fc3-ac67-4dfa-b0a3-bcba3069d17b


### 13. **Delete Infrastructure (Optional)**

Please delete below mentioned services
    
    - Workbench
    - Storage Bucket
    - Delete Pipelines
    - Artifact Repository created in Artifact Registry
    - Undeploy the model from endpoint
    - Delete the endpoint
    - Delete model from model registry
    - Delete data from Metadata management
    - Delete Cloud Alert
    - Delete Pub Sub Topics
    - Delete Dataflow jobs
    - Delete model monitoring

#### 🎥 ***Demo Video***
https://github.com/user-attachments/assets/5b765fcd-16e9-42e8-a9cf-2baf134e9f07


## Implementation
To test the code we need to do the following:

    1. Copy the repository in Cloud SDK using below command:
    git clone https://github.com/adityasolanki205/Unified-ETL-DWH-MLOps-Pipeline.git
    
    2. Create a Storage Bucket by the name 'demo_bucket_kfl' in asia-south1 and two sub folders Temp and Stage.
    
    3. Copy the data file in the cloud Bucket using the below command
    cd ML_Pipeline_using_Kubeflow
    gsutil cp german_data.csv gs://demo_bucket_kfl/
    
    4. Create a Dataset in asia-east1 by the name GermanCredit
    
    5. Create a table in GermanCredit dataset by the name GermanCreditTable. 
        Schema is present at the starting of batch-pipeline.py

    6. Create a table in GermanCredit dataset by the name GermanCreditTable-streaming. 
        Schema is present at the starting of ml-streaming-pipeline-endpoint.py

    7. Create Pub Sub Topic by the name german_credit_data and Model_Monitoring
    
    8. Install Apache Beam on the SDK using below command
    pip3 install apache_beam[gcp]
    
    9. Command to run Batch job:
     python3 batch-pipeline.py \
     --runner DataFlowRunner \
     --project solar-dialect-264808 \
     --temp_location gs://demo_bucket_kfl/Temp \
     --staging_location gs://demo_bucket_kfl/Stage \
     --input gs://demo_bucket_kfl/german_data.csv \
     --region asia-south1 \
     --job_name germananalysis

    10. Run the file training_pipeline.ipynb/training_pipeline.py in workbench. This will create a json file.
    
    11. Run the run_pipeline.ipynb file
     
    12. Verify of all the artifacts are created.
    
    13. The Streaming pipeline will run with below configuration only. To configure environment run commands present in update_python.ipynb
        Python 3.11, apache-beam[gcp]==2.64.0

    14. Run the pipeline using:
    python3 ml-streaming-pipeline-endpoint.py \
      --runner DataFlowRunner \
      --project solar-dialect-264808 \
      --bucket_name demo_bucket_kfl \
      --temp_location gs://demo_bucket_kfl/Temp \
      --staging_location gs://demo_bucket_kfl/Stage \
      --region asia-south1 \
      --job_name ml-stream-analysis \
      --input_subscription projects/solar-dialect-264808/subscriptions/german_credit_data-sub \
      --input_topic projects/solar-dialect-264808/topics/german_credit_data \
      --save_main_session \
      --setup_file ./setup.py \
      --max_num_workers 1 \
      --streaming
      
    15. Open one more tab in cloud SDK and run below command 
    cd ML-Streaming-pipeline-using-Dataflow
    python3 publish_to_pubsub.py

    16. Goto Model Monitoring and setup model monitoring for output drift detection

    17. Create a Alerting policy for Model output drift deviation with threshold as 0.3 and select notification 
    channel as model_monitoring topic in pub sub. 

    18. Create a Cloud Run functions that listens for Pub Sub and triggers retraining pipeline. So when cloud alerting
    triggers a message to Pub sub, Cloud Run Functions gets invoked and starts retraining.

## Credits
1. Akash Nimare's [README.md](https://gist.github.com/akashnimare/7b065c12d9750578de8e705fb4771d2f#file-readme-md)
2. [Building a Kubeflow Training Pipeline on Google Cloud: A Step-by-Step Guide](https://medium.com/@rajmudigonda893/building-a-kubeflow-training-pipeline-on-google-cloud-a-step-by-step-guide-761a6b0eb197)
3. [Deploy ML Training Pipeline Using Kubeflow](https://medium.com/@kavinduhapuarachchi/deploy-ml-training-pipeline-using-kubeflow-19d52d22f44f)
4. [A Beginner’s Guide to Kubeflow on Google Cloud Platform](https://medium.com/@vishwanath.prudhivi/a-beginners-guide-to-kubeflow-on-google-cloud-platform-5d02dbd2ec5e)
5. [MLOps 101 with Kubeflow and Vertex AI](https://medium.com/google-cloud/mlops-101-with-kubeflow-and-vertex-ai-61f6f5489fa8)
