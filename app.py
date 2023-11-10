import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain import PromptTemplate
import pickle

load_dotenv()



system_prompt = "You are an friendly ai assistant that help users find the most relevant and accurate answers to their questions based on the documents you have access to. When answering the questions, mostly rely on the info in documents."

query_wrapper_prompt = '''
you are an friendly ai assistant that help users find the most relevant and accurate answers to their questions based on the documents you have access to. When answering the questions, only rely on the info in documents.
The document information is below.
---------------------
{context_str}
---------------------
Using the document information and mostly relying on it,
answer the query.
Query: {query_str}
if you don't know the answer, say "I don't know." but dont take information from outside the document.
Answer:
'''


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    print("Query:", query)
    # print("History:", history)
    # print('chain', chain)

    result = chain({"question": query, "chat_history": history})
    prompt = PromptTemplate.from_template(template=query_wrapper_prompt)
    print("Answer:", result["answer"])
    if "I don't know." in result["answer"]:
        return "I don't know."
    else:
        prompt_formatted_str: str = prompt.format(context_str=result["answer"], query_str=query)
        llm = OpenAI()

        # make a prediction
        prediction = llm.predict(prompt_formatted_str)
        history.append((query, prediction))

        return prediction

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    load_dotenv()
    # Create llm
    llm = OpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                                                 memory=memory)
                                                 
    return chain




def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("AnuragGPT :books:")
    # Initialize Streamlit
    # st.sidebar.title("Document Processing")
    # uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if os.path.exists('traningdata.pkl'):
        with open(f"traningdata.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        chain = create_conversational_chain(VectorStore)

        
        display_chat_history(chain)

    # elif uploaded_files:
    #     text = []
    #     for file in uploaded_files:
    #         file_extension = os.path.splitext(file.name)[1]
    #         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    #             temp_file.write(file.read())
    #             temp_file_path = temp_file.name

    #         loader = None
    #         if file_extension == ".pdf":
    #             loader = PyPDFLoader(temp_file_path)
    #         elif file_extension == ".docx" or file_extension == ".doc":
    #             loader = Docx2txtLoader(temp_file_path)
    #         elif file_extension == ".txt":
    #             loader = TextLoader(temp_file_path)

    #         if loader:
    #             text.extend(loader.load())
    #             os.remove(temp_file_path)

    #     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    #     text_chunks = text_splitter.split_documents(text)

    #     # Create embeddings
    #     embeddings = OpenAIEmbeddings()
    #     # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
    #     #                                    model_kwargs={'device': 'cpu'})

    #     # Create vector store
    #     vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    #     # Create the chain object
    #     chain = create_conversational_chain(vector_store)

        
    #     display_chat_history(chain)

if __name__ == "__main__":
    main()