import json
import urllib.parse
import boto3
import pandas as pd
from io import StringIO
import re
import time

print('Loading function')

s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

def fix_titles(title):
    # Define a pattern to match HTML tags
    html_tags_pattern = re.compile(r'<.*?>')

    # Replace HTML tags with an empty string
    cleaned_title = re.sub(html_tags_pattern, '', title)

    return cleaned_title

def preprocess_data(df):
    """
    Preprocess the DataFrame.
    This function Pre Processes the dataframe

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # Example preprocessing: Drop any rows with missing values
    df = df.dropna()
    df = df.drop(columns=['id'])
    df = df.drop_duplicates()
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['title'] = df['title'].apply(fix_titles)
    df["claps"] = df["claps"].fillna(0)
    df["subtitle"] = df["subtitle"].fillna(df["title"])
    df['article'] = df['title'] + df['subtitle']
    return df


# 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04

#
# 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.14.1-cpu-py310-ubuntu20.04-sagemaker

# Pick the right image for training and task using this link https://github.com/aws/deep-learning-containers/blob/master/available_images.md
def trigger_sagemaker_training(bucket, processed_key):
    training_job_name = 'content-recommender-training-job-' + str(int(time.time()))
    response = sagemaker.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification={
            'TrainingImage': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker',  # Replace with your custom training image
            'TrainingInputMode': 'File'
        },
        RoleArn='arn:aws:iam::765477734195:role/AmazonSageMaker-ExecutionRole',  # Replace with your SageMaker execution role
        InputDataConfig=[
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{bucket}/{processed_key}',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv'
            }
        ],
        OutputDataConfig={
            'S3OutputPath': f's3://{bucket}/output/'
        },
        ResourceConfig={
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 10
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 7200
        },
        HyperParameters = {
            'sagemaker_program': 'train.py',
            'sagemaker_requirement': 'requirements.txt',
            'sagemaker_submit_directory': f's3://{bucket}/scripts/full_training_data.tar.gz'
        }
    )
    return response

def lambda_handler(event, context):
    # print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    # Skip processing if the file is already in the processed folder
    if key.startswith('processed/'):
        return {
            'statusCode': 200,
            'body': json.dumps('Skipping already processed file.')
        }
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        # Read the CSV content
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))

        # Preprocess the data
        processed_df = preprocess_data(df)

        # Save the processed data back to S3
        processed_csv_buffer = StringIO()
        processed_df.to_csv(processed_csv_buffer, index=False)

        processed_key = f'processed/{key.split("/")[-1]}'
        s3.put_object(Bucket=bucket, Key=processed_key, Body=processed_csv_buffer.getvalue())

        # Trigger SageMaker training
        sagemaker_response = trigger_sagemaker_training(bucket, 'processed/')

        print("CONTENT TYPE: " + response['ContentType'])
        return {
            'statusCode': 200,
            'body': json.dumps('SageMaker training job triggered successfully'),
            'sagemaker_response': sagemaker_response
        }
    except Exception as e:
        print(e)
        print(
            'Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(
                key, bucket))
        raise e
