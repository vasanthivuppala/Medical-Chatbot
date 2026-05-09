
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)
load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

#Make sure these keys are available globally for other libraries
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

embeddings=download_hugging_face_embeddings()

index_name="medical-chatbot"

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})
chatModel=ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
   [   
    ("system", system_prompt),
    ("human", "{input}"),
   ]
)

question_answer_chain=create_stuff_documents_chain(chatModel,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():

    msg = request.form["msg"]

    response = rag_chain.invoke({
        "input": msg
    })

    return response["answer"]



if __name__ == "__main__":
    app.run(debug=True)