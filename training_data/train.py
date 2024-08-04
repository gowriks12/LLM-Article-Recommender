import subprocess
import sys
import pandas as pd
import os
import boto3
import numpy as np
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def install_dependencies():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "/opt/ml/code/requirements.txt"])


def preprocess_data(data_path):
    documents = []
    loader = CSVLoader(file_path=data_path, source_column="title",encoding="utf-8")
    documents.extend(loader.load())
    # print(documents)
    return documents


def embed_articles(documents):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(documents, embeddings)
    return embeddings, vectorstore


def save_faiss_index(vector_store_path, vectorstore):
    vectorstore.save_local(vector_store_path)

def upload_directory_to_s3(directory_path, bucket_name, s3_prefix):
    s3_client = boto3.client('s3')
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            s3_key = os.path.join(s3_prefix, os.path.relpath(file_path, directory_path))
            s3_client.upload_file(file_path, bucket_name, s3_key)
            print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")


def upload_to_s3(file_path, bucket, key):
    s3 = boto3.client('s3')
    s3.upload_folder(file_path, bucket, key)


if __name__ == '__main__':
    install_dependencies()
    input_data_path = '/opt/ml/input/data/training/medium_data.csv'
    model_dir = '/opt/ml/model'
    # faiss_index_file = os.path.join(model_dir, 'faiss_index.index')
    vector_store_path = os.path.join(model_dir, 'faiss_vector_store')

    documents = preprocess_data(input_data_path)
    embeddings, vectorstore = embed_articles(documents)
    save_faiss_index(vector_store_path, vectorstore)

    # Ensure the model is saved correctly
    # torch.save(embeddings, os.path.join(model_dir, 'embeddings.pt'))

    # Upload the FAISS index to S3
    bucket = 'llm-rec-sys-1901'
    # key = 'faiss/faiss_index.index'
    # upload_to_s3(faiss_index_file, bucket, key)
    s3_prefix = 'faiss_vector_store'
    # upload_to_s3(vector_store_path, bucket, key)
    upload_directory_to_s3(vector_store_path, bucket, s3_prefix)
