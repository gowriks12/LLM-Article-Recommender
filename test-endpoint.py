import requests
import pandas as pd
import json
def get_recommendations(query, k=20):
    lambda_function_url = "your-lambda-url"
    req = {"text": query, "k": k}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=lambda_function_url, json=req,headers=headers)
    resp_dict = {}
    recoms = (response.json())
    for k in recoms:
        v = recoms[k]
        values = v.split('\n')
        for val in values:
            if val.split(': ')[0] in resp_dict:
                resp_dict[val.split(': ')[0]].append(val.split(': ')[1])
            else:
                resp_dict[val.split(': ')[0]] = [val.split(': ')[1]]

    # print("post",resp_dict)
    recom_df = pd.DataFrame.from_dict(resp_dict)
    # print(df)
    # recoms = pd.DataFrame(response.json())
    # self.df = recoms
    return recom_df

print(get_recommendations(query="what is chatGPT?"))