# import boto3
import sagemaker
import json

#
sagemaker_session = sagemaker.Session()
#
# ENDPOINT_NAME='llm-rec-sys-endpoint-1'
# runtime=boto3.client('runtime.sagemaker', region_name='us-east-1')
#
# # instruction = "What is chatGPT"
#
# response = runtime.invoke_endpoint(
#     EndpointName=ENDPOINT_NAME,
#     ContentType='application/json',
#     Body=json.dumps({'text': "What is chatGPT"})
# )
#
# result = json.loads(response['Body'].read().decode())
# print(result)


from sagemaker.predictor import Predictor
sagemaker_vector_store = sagemaker.predictor.Predictor('faiss-endpoint-1721914626')
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
# predictor = Predictor(endpoint_name='pytorch-inference-2024-07-21-02-50-10-131')
# inp = {"query": "Machine learning in healthcare"}
#
# # Convert the input data to JSON string
# json_input = json.dumps(inp)
#
# # Make the prediction
# response = predictor.predict(json_input)
#
# # response = predictor.predict(inp)
# print(response)
