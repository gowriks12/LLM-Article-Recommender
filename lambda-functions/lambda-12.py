import boto3
import json

sagemaker_client = boto3.client('sagemaker')
sagemaker_runtime = boto3.client('sagemaker-runtime')


def get_latest_faiss_endpoint():
    # List all endpoints
    response = sagemaker_client.list_endpoints()
    endpoints = response['Endpoints']

    # Filter endpoints that start with 'faiss'
    faiss_endpoints = [ep for ep in endpoints if ep['EndpointName'].startswith('faiss-endpoint')]

    # Sort endpoints by CreationTime
    faiss_endpoints.sort(key=lambda ep: ep['CreationTime'], reverse=True)

    if faiss_endpoints:
        return faiss_endpoints[0]['EndpointName']
    else:
        raise Exception("No endpoint starting with 'faiss' found")


def lambda_handler(event, context):
    # Prepare the payload
    query = event['text']
    k = event['k']
    payload = {
        'text': query,
        'k': k
    }
    endpoint_name = get_latest_faiss_endpoint()
    print(endpoint_name)
    # Call the SageMaker endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,  # replace with your endpoint name
        Body=json.dumps(payload),
        ContentType='application/json'
    )

    # Parse the response
    result = json.loads(response['Body'].read().decode())

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
