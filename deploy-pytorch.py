import sagemaker
from sagemaker.pytorch import PyTorchModel
import time
import json

# Assume you have already trained and saved your model artifacts in S3
model_artifact = 's3://llm-rec-sys-1901/output/content-recommender-training-job-1721872511/output/model.tar.gz'
role = 'arn:aws:iam::765477734195:role/AmazonSageMaker-ExecutionRole'

image = sagemaker.image_uris.retrieve(
    framework='pytorch',
    region='us-east-1',
    image_scope='inference',
    version='1.12',
    instance_type='ml.m5.2xlarge'
)
print(image)

model_name = f'faiss-vs-{int(time.time())}'
faiss_model_sm = sagemaker.model.Model(
    model_data=model_artifact,
    image_uri=image,
    role=role,
    entry_point='inference.py',
    source_dir='s3://llm-rec-sys-1901/scripts/inference.tar.gz',
    name=model_name
)
print(faiss_model_sm)

endpoint_name = f'faiss-endpoint-{int(time.time())}'
faiss_model_sm.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name=endpoint_name,
    wait=True,
)

sagemaker_vs = sagemaker.predictor.Predictor(endpoint_name)
print(sagemaker_vs)

print("testing endpoint")

sagemaker_vector_store = sagemaker.predictor.Predictor(endpoint_name)
assert sagemaker_vector_store.endpoint_context().properties['Status'] == 'InService'

print(sagemaker_vector_store)

payload = json.dumps({
    "text": "what is a chatGPT",
    "k": 10,
})

out = sagemaker_vector_store.predict(
    payload,
    initial_args={"ContentType": "application/json", "Accept": "application/json"}
)
out = json.loads(out)
print(out)
