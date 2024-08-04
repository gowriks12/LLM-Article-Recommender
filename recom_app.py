import pickle
import pandas as pd
import streamlit as st
from streamlit import session_state as session
import types
from recommendations import Recommendations

recommender = Recommendations()

# def my_hash_func(func):
#     # Custom hash function for functions
#     return hash(func.__code__)

# @st.cache_data(persist=True, show_spinner=False, hash_funcs={types.FunctionType: my_hash_func})
# def load_data():
#     """
#     load and cache data
#     :return: tfidf data
#     """
#     recom_df = pd.read_csv("data/recom_df.csv", index_col=0)
#     article_list = recom_df.index.tolist()
#     return recom_df, article_list

# recom_df, article_list = load_data()

top_publication_content, trending_articles, top_quick_reads, recommended = None, None, None, None

st.title("""
Medium Article Recommendation System
This is an Content Based Recommender System which uses LLM to recommend articles similar to user query.
 """)

st.text("")
st.text("")
st.text("")
st.text("")

# session.options = st.selectbox(label="Select Article", options=article_list)
query = st.text_input("Query", placeholder="What do you want recommendations for?...")  #or st.button("Recommend")
# st.write("You Selected: ",session.options)
# print(session.options)

st.text("")
st.text("")

slider_count = st.slider(label="Article Count", min_value=5, max_value=50)

st.text("")
st.text("")

buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

is_clicked = col1.button(label="Recommend")

if is_clicked:
    # dataframe = recommender.recommend_articles(recom_df=recom_df, article = session.options)
    top_publication_content, trending_articles, top_quick_reads, recommended = recommender.recommend_articles(query=query, k=slider_count)

st.text("")
st.text("")
st.text("")
st.text("")

# st.title("Recommendations based on selections")
st.subheader('Recommendations based on selections', divider='red')
if recommended is not None:
    print(recommended)
    st.data_editor(
        recommended,
        column_config={
            "url": st.column_config.LinkColumn(
                "Link", display_text="URL"
            ),
        },
        hide_index=False,
    )

# st.title("Top Publication Content Recommendations")
st.subheader('Top Publication Content Recommendations', divider='red')
if top_publication_content is not None:
    top_publication_content = top_publication_content[["title", "url", "publication", "claps"]]
    print(top_publication_content)
    st.data_editor(
        top_publication_content,
        column_config={
            "url": st.column_config.LinkColumn(
                "Link", display_text="URL"
            ),
        },
        hide_index=True,
    )
    # st.table(top_publication_content['title'])

# st.title("Trending Content Recommendations")
st.subheader('Trending Content Recommendations', divider='red')
if trending_articles is not None:
    trending_articles = trending_articles[["title", "url", "publication", "claps"]]
    print(trending_articles)
    st.data_editor(
        trending_articles,
        column_config={
            "url": st.column_config.LinkColumn(
                "Link", display_text="URL"
            ),
        },
        hide_index=True,
    )
    # st.table(trending_articles['title'])

# st.title("Quick Read Recommendations")
st.subheader('Quick Read Recommendations', divider='red')
if top_quick_reads is not None:
    top_quick_reads = top_quick_reads[["title", "url", "publication", "claps"]]
    print(top_quick_reads)
    st.data_editor(
        top_quick_reads,
        column_config={
            "url": st.column_config.LinkColumn(
                "Link", display_text="URL"
            ),
        },
        hide_index=True,
    )
    # st.table(top_quick_reads['title'])
