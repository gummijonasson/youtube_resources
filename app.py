import os 
from dotenv import load_dotenv, find_dotenv

from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import textwrap
from metaphor_python import Metaphor

import streamlit as st 


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
METAPHOR_API_TOKEN = os.environ.get("METAPHOR_API_TOKEN")

metaphor = Metaphor(METAPHOR_API_TOKEN)

st.title(':tv: YouTube Learning Resource Summarizer')
prompt = st.text_input('Plug in your YouTube video URL here:') 

repo_id = "tiiuae/falcon-7b-instruct"
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)


combine_prompt = """
Summarize what this youtube video is about in only 3-4 sentences of the following:
"{text}"
SUMMARY:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

def find_similar(url):
    response = metaphor.find_similar(url, num_results=3, include_domains=["youtube.com"])
    url = []
    title = []
    for result in response.results:
        url.append(result.url)
        title.append(result.title)
    return url, title

def get_video_summary(urls):
    video_url = urls
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    docs = text_splitter.split_documents(transcript)

    chain = load_summarize_chain(falcon_llm, combine_prompt=combine_prompt_template,chain_type="map_reduce", verbose=False)

    output_summary = chain.run(docs)
    wrapped_text = textwrap.fill(output_summary, width=100, break_long_words=False, replace_whitespace=False)
    return wrapped_text

   

if prompt:
    urls, titles = find_similar(prompt)
    
    for video in urls:
        title = titles[urls.index(video)]
        summary = get_video_summary(video)
        st.write(title + ', Link to video: ' + video)
        st.write(summary)
    
