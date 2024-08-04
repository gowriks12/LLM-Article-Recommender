import json
import boto3
import time
import tarfile
import os

s3 = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker')


def repackage_model(model_s3_key, inference_code_key, bucket, training_job_name):
    download_model_path = '/tmp/model.tar.gz'
    download_inference_code_path = '/tmp/inference.tar.gz'
    extract_model_path = '/tmp/model'
    extract_inference_code_path = '/tmp/inference_code'
    repackaged_model_path = '/tmp/repackaged_model.tar.gz'

    # Download the model tarball from S3
    s3.download_file(bucket, model_s3_key, download_model_path)

    # Extract the model tarball
    with tarfile.open(download_model_path, 'r:gz') as tar:
        tar.extractall(path=extract_model_path)

    # Download the inference code tarball from S3
    s3.download_file(bucket, inference_code_key, download_inference_code_path)

    # Extract the inference code tarball
    with tarfile.open(download_inference_code_path, 'r:gz') as tar:
        tar.extractall(path=extract_inference_code_path)

    # Repackage the model with the inference code
    with tarfile.open(repackaged_model_path, 'w:gz') as tar:
        for root, _, files in os.walk(extract_model_path):
            for file in files:
                tar.add(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), extract_model_path))
        for root, _, files in os.walk(extract_inference_code_path):
            for file in files:
                tar.add(os.path.join(root, file),
                        arcname=os.path.relpath(os.path.join(root, file), extract_inference_code_path))

    # Upload the repackaged model to S3
    repackaged_model_s3_key = f'output/{training_job_name}/output/repackaged_model.tar.gz'
    s3.upload_file(repackaged_model_path, bucket, repackaged_model_s3_key)

    return f's3://{bucket}/{repackaged_model_s3_key}'


def get_latest_training_job_name():
    response = sagemaker_client.list_training_jobs(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    latest_training_job = response['TrainingJobSummaries'][0]['TrainingJobName']
    return latest_training_job


def lambda_handler(event, context):
    training_job_name = get_latest_training_job_name()
    # training_job_name = "content-recommender-training-job-1721883070"
    model_name = f'faiss-vs-{int(time.time())}'
    endpoint_config_name = f'{model_name}-endpoint-config'
    endpoint_name = f'faiss-endpoint-{int(time.time())}'
    bucket = 'llm-rec-sys-1901'
    region = 'us-east-1'
    execution_role_arn = 'arn:aws:iam::765477734195:role/AmazonSageMaker-ExecutionRole'
    # container_image = '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.3.0-cpu-py311-ubuntu20.04-sagemaker'  # e.g., '123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:latest'
    container_image = '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12-cpu-py38'  # e.g., '123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:latest'

    #
    # Construct the S3 model path
    model_data_url = f's3://{bucket}/output/{training_job_name}/output/model.tar.gz'
    model_key = f'output/{training_job_name}/output/model.tar.gz'
    print(model_key)
    # model_data_url = 's3://llm-rec-sys-1901/output/content-recommender-training-job-1721883070/output/model.tar.gz'
    inference_code = 's3://llm-rec-sys-1901/scripts/inference.tar.gz'
    inf_key = 'scripts/inference.tar.gz'

    # Path to your inference code
    entry_point = 'inference.py'

    rmodel_path = repackage_model(model_key, inf_key, bucket, training_job_name)
    print(rmodel_path)

    # Create the model
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': container_image,
            'ModelDataUrl': rmodel_path,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': 's3://llm-rec-sys-1901/scripts/inference.tar.gz',
                'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                'SAGEMAKER_REGION': 'us-east-1',
                'MMS_DEFAULT_RESPONSE_TIMEOUT': '500'
            }
        },
        ExecutionRoleArn=execution_role_arn
    )

    # Create the endpoint configuration
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m4.xlarge',
                'InitialVariantWeight': 1
            }
        ]
    )

    # Create the endpoint
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    return {
        'statusCode': 200,
        'body': json.dumps('Endpoint creation initiated successfully!')
    }
