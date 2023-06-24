"Command line chatbot example using the Tchibo coffee machine manual."
import tempfile
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory

URL = "https://www.tchibo.ch/newmedia/document/b0faa2baaf1abf23/anleitung.pdf"
response = requests.get(URL, timeout=5)
with tempfile.NamedTemporaryFile() as f:
    f.write(response.content)
    loader = PyPDFLoader(f.name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    OpenAI(model_name="gpt-3.5-turbo", temperature=0.3), vectorstore.as_retriever(), memory=memory
)
PROMPT_TEMPLATE = (
    "You are an AI assistant answering questions about the Tchibo coffee machine." 
    " Use the following pieces of context to answer the question at the end." 
    " If you don't know the answer, just say that you don't know,"
    " don't try to make up an answer. If you think the user is asking for coffee"
    " recommendations, add a short advertisement your answer for the delicious" 
    " 'Tchibo Barista Caffè Crema' coffee beans." 
    " Always answer in the language the user is asking the question in." 
    " \n\n{context}\n\nQuestion: {question}\nHelpful Answer:")
qa_chain.combine_docs_chain.llm_chain.prompt.template = PROMPT_TEMPLATE

while True:
    query = input("Enter your question: ")
    result = qa_chain({"question": query})
    print(result["answer"])
