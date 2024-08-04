import subprocess
import sys
import os
import json

def install_packages():
    # Install required packages from requirements.txt
    requirements_path = os.path.join(os.path.dirname(__file__), '../requirements.txt')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])

install_packages()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def model_fn(model_dir):
    """
    Load the FAISS vector store and huggingface model into memory.
    """

    # load huggingface embedding model
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # model_kwargs = {'device': 'cpu'}
    # encode_kwargs = {'normalize_embeddings': False}
    # hf = HuggingFaceBgeEmbeddings(
    #     model_name=model_name,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs,
    #     query_instruction="Represent this sentence for searching relevant documents: ",
    # )
    hf = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # load vector store
    vs_path = os.path.join(model_dir, 'faiss_vector_store')
    vs = FAISS.load_local(vs_path, hf, allow_dangerous_deserialization=True)
    return vs


def input_fn(request_body, request_content_type):
    """
    Takes in request and transforms it to necessary input type - in this case we use json inputs
    """
    if request_content_type == 'application/json':
        request_body = json.loads(request_body)
    else:
        raise ValueError("Content type must be application/json")
    return request_body


def predict_fn(input_data, model):
    """
    SageMaker model server invokes `predict_fn` on the return value of `input_fn`.

    This function returns the similarity search results
    """
    vs = model
    results = vs.similarity_search(input_data['text'], input_data['k'])
    return results


def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    We wrap the output text into a json format here.
    """
    out_dict = dict(zip(range(len(predictions)), range(len(predictions))))
    for ind in range(len(predictions)):
        out_dict[ind] = predictions[ind].page_content
    return json.dumps(out_dict)