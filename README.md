# Steps to Run the Recommendation Endpoint using LLM

## 1. Upload data/medium_data.csv to S3 bucket llm-rec-sys-1901
## 2. Creating tar files
- cd training_data
- tar -cvzf training_data.tar.gz requirements.txt train.py
- cd ..
- cd inference
- tar -cvzf inference.tar.gz inference.py requirements.txt
- cd ..
Upload training_data.tar.gz and inference.tar.gz to S3 bucket llm-rec-sys-1901/scripts
## 3. Create a Lambda Function using the code in lambda-11.py
 - Use S3 bucket trigger
 - add pandas Layer
 - use lambda-test-event.json as test event to trigger the lambda function using medium_data.csv
 - This lambda function preprocesses the medium_data.csv and uploads it to s3 bucket and triggers
a sagemaker training job.
## 4. Training Job created 
 - Training job uses the training_data.tar.gz as the script to carry out training.
 - Check for successful completion of training job
## 5. Create Event Rule on Event Bridge
 - Create a rule to trigger lambda-deploy-1 lambda function when training job is complete
 - lambda-deploy-1 function loads the new training model, repackages with the inference code 
and deploys the endpoint on sagemaker
## 6. Invoke Endpoint
 - Invoke the latest created endpoint using lambda-12 function
## 7. Streamlit Application
 - Streamlit Application connects to the lambda function using the function url of the lambda function
 - Start up the streamlit app by running "streamlit run recom_app.py"
 - Run queries and get recommendations
## 8. Clean Up
 - Delete endpoint and model on AWS console or using aws cli

# Demo GIF
![Logo](https://github.com/gowriks12//blob/develop/static/Demo-GIF-1.gif)