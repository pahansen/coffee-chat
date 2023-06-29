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


def coffee_chat(
    document_url: str = "https://www.tchibo.ch/newmedia/document/b0faa2baaf1abf23/anleitung.pdf",
    create_db: bool = False,
):
    embeddings = OpenAIEmbeddings()
    if create_db:
        response = requests.get(document_url, timeout=5)
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(response.content)
            loader = PyPDFLoader(temp_file.name)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents, embeddings, persist_directory="chroma_db"
            )
            vectorstore.persist()
    else:
        vectorstore = Chroma(
            persist_directory="chroma_db", embedding_function=embeddings
        )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        OpenAI(model_name="gpt-3.5-turbo", temperature=0.3),
        vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    prompt_template = (
        "You are an AI assistant answering questions about the Tchibo coffee machine."
        " Use the following pieces of context to answer the question at the end."
        " If you don't know the answer, just say that you don't know,"
        " don't try to make up an answer. If you think the user is asking for coffee"
        " recommendations, add a short advertisement your answer for the delicious"
        " 'Tchibo Barista Caffè Crema' coffee beans."
        " Always answer in the language the user is asking the question in."
        " \n\n{context}\n\nQuestion: {question}\nHelpful Answer:"
    )
    qa_chain.combine_docs_chain.llm_chain.prompt.template = prompt_template

    while True:
        query = input("Enter your question: ")
        result = qa_chain({"question": query})
        answer = result["answer"]
        answer += "\n"
        for source_doc in result["source_documents"]:
            source = source_doc.metadata["source"]
            answer += f"[{source}]"
        print(answer)


if __name__ == "__main__":
    coffee_chat(create_db=False)
