import pickle
import pandas as pd
# import streamlit as st
# from streamlit import session_state as session
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
import requests


class Recommendations:
    # def __init__(self):
        # self.df = pd.read_csv("data/medium_data.csv", index_col="id")
        # self.df = self.df.drop_duplicates()

    def get_recommendations(self, query, k=20):
        lambda_function_url = "https://sajjk4n2rwes6bvmdbl7ndu5gi0boklo.lambda-url.us-east-1.on.aws/"
        req = {"text": query, "k": k}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url=lambda_function_url, json=req, headers=headers)
        resp_dict = {}
        recoms = (response.json())
        print(recoms)
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
        # recom_df.drop(["id"])
        recom_df = recom_df.drop_duplicates()
        # recoms = pd.DataFrame(response.json())
        self.df = recom_df
        print(recom_df.shape)
        return recom_df


    def preprocess_data(self):
        self.df['date'] = pd.to_datetime(self.df['date'], format="%Y-%m-%d")
        # self.df['id'] = self.df['id'].astype(int)
        self.df['claps'] = pd.to_numeric(self.df['claps'])
        self.df['responses'] = self.df['responses'].astype(int)
        self.df['reading_time'] = self.df['reading_time'].astype(int)
        self.df = self.df.drop_duplicates()
        print(self.df.shape)
        return self.df


    def top_content(self):
        pub_popularity = self.df.groupby('publication')[['claps', 'responses']].mean().round().astype(int).sort_values(
            by='claps', ascending=False)
        top_three_publications = pub_popularity['claps'].nlargest(3).index
        channels = top_three_publications.tolist()
        top_articles = pd.DataFrame()  # Initialize an empty DataFrame to store top articles

        for channel in channels:
            cont = self.df[self.df['publication'] == channel]
            top_n_articles = cont.nlargest(3, 'claps')  # Select top 3 articles for the channel
            top_articles = pd.concat([top_articles, top_n_articles])  # Concatenate with previous top articles
        top_articles = top_articles.drop_duplicates()
        return top_articles

    # print(top_content(df))

    def trending_article(self):
        latest_date = self.df['date'].max()
        latest_week = latest_date - pd.Timedelta(days=6)
        latest_articles = self.df[self.df['date'] >= latest_week]
        top_three = self.df.loc[latest_articles['claps'].nlargest(3).index]
        top_three_trending = top_three['title'].tolist()
        return top_three

    # print(trending_article(df))

    def popular_quick_reads(self):
        avg_reading_time = self.df['reading_time'].mean()
        quick_reads = self.df[self.df['reading_time'] <= avg_reading_time]
        quick_reads_df = self.df.loc[quick_reads['claps'].nlargest(3).index]
        #     popular_quick_reads = quick_reads_df['title'].tolist()
        return quick_reads_df


    def recommend_articles(self, query, k):
        print("In recommend articles---------------")
        print("Data Preprocessed-------------------")
        recom_df = self.get_recommendations(query, k)
        # a = self.df[self.df['title'] == article[0]]
        # art = recom_df.loc[article]
        # print(art)
        recom_df = self.preprocess_data()
        top_publication_content = self.top_content()
        trending_articles = self.trending_article()
        top_quick_reads = self.popular_quick_reads()
        # recommended = self.content_based_recommendation(recom_df=recom_df, article=art, count=count)
        # recomms = recommended.loc[1:]
        # recommended = recommended.drop(recommended.index[0])
        recommended = recom_df[["title","url","publication","claps"]]
        print(recommended)

        return top_publication_content, trending_articles, top_quick_reads, recommended


if __name__=="__main__":
    recommender = Recommendations()
    count =20
    # article = ["How ChatGPT Works: The Model Behind The\xa0Bot"]
    query = "What is ChatGPT?"
    top_publication_content, trending_articles, top_quick_reads, recommended = recommender.recommend_articles(query, count)
    print("-----------------------------------------------")
    print("----------------Recommendations----------------")
    print("-----------------------------------------------")
    print(" ")
    print("----Top Publication Content Recommendations----")
    print(top_publication_content['title'])
    print("-------Trending Content Recommendations--------")
    print(trending_articles['title'])
    print("-----------Quick Read Recommendations----------")
    print(top_quick_reads['title'])
    print("----Because You read ", query, "----")
    print(recommended.index)
    print("-----------------------------------------------")

