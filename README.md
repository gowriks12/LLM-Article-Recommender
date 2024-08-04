# Steps to Run the Recommendation Endpoint using LLM

## 1. Upload medium_data.csv to S3 bucket llm-rec-sys-1901
## 2. Creating tar files
- tar -cvzf training_data.tar.gz requirements.txt train.py
- tar -cvzf inference.tar.gz inference.py requirements.txt
Upload training_data.tar.gz and inference.tar.gz to S3 bucket llm-rec-sys-1901/scripts
## 3. Create a Lambda Function using the code in lambda-11.py
 - Use S3 bucket trigger
 - add pandas Layer
 - use lambda-test-event.json as test event to trigger the lambda function using medium_data.csv
## 4. Training Job created 
 - Check for successful completion
## 5. Create Event Rule on Event Bridge
 - Create a rule to trigger lambda-deploy-1 lambda function when training job is complete
 - lambda-deploy-1 function loads the new training model, repackages with the inference code 
and deploys the endpoint on sagemaker
## 6. Invoke Endpoint
 - Invoke the latest created endpoint using lambda-12 function
## 6. Clean Up
 - Delete endpoint and model