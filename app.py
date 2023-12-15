import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
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
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain import PromptTemplate
import pickle
import pyrebase
load_dotenv()


firebaseConfig = {
  'apiKey': "",
  'authDomain': "",
  'projectId': "",
  'storageBucket': "",
  'messagingSenderId': "",
  'appId': "",
  'measurementId': "",
  "databaseURL":''
}


# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

def loginn():
    placeholder = st.empty()
    placeholder0 = st.empty()
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    placeholder3 = st.empty()
    placeholder4 = st.empty()
    placeholder5 = st.empty()
    placeholder6 = st.empty()
    placeholder7 = st.empty()
    placeholder.title("AnuragGPT :books:")
    choice = placeholder1.selectbox('Login / Signup', ['Login', 'Sign up'])

    if choice == 'Login':
        # Login form
        email = placeholder2.text_input('Email', placeholder='Enter your email')
        password = placeholder3.text_input('Password', type='password', placeholder='Enter your password')

        if placeholder4.checkbox('Login'):
            try:
                # Login user
                user = auth.sign_in_with_email_and_password(email, password)
                print(user)
                placeholder5.success(user['displayName'])
                

              

                # Display success message
                placeholder6.success('Logged in successfully!')
                placeholder.empty()
                placeholder1.empty()
                placeholder2.empty()
                placeholder3.empty()
                placeholder4.empty()
                placeholder5.empty()
                placeholder6.empty()
                return True
            except Exception as e:
                st.error(e)
    elif choice == 'Sign up':
    # Signup form
        email = st.text_input('Email', placeholder='Enter your email')
        password = st.text_input('Password', type='password', placeholder='Enter your password')
        username = st.text_input('Username', placeholder='Enter your username')

        if st.button('Sign Up'):
            try:
                # Create user
                user = auth.create_user_with_email_and_password(email, password)
                st.success(user)
                data = {"name":username,"email":email,'username':username,'data':'','lastlogin':[]}
                data['lastlogin'].append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                db.child("users").child(user['localId']).set(data)
                # Display success message
                st.success('Account created successfully! Please check your email for the verification link.')
            except Exception as e:
                st.error(e)

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

    if True:
        # with open(f"traningdata.pkl", "rb") as f:
        #     VectorStore = pickle.load(f)
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        db3 = Chroma(persist_directory="./anuragweb", embedding_function=embedding_function)

        chain = create_conversational_chain(db3)


        display_chat_history(chain)


if __name__ == "__main__":
    sc = loginn()
    if sc:
        main()
