import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.prompts import PromptTemplate
import logging
import time 


st.set_page_config(page_title="Insurance Policy Chatbot", page_icon="🛡️", layout="wide")


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.") 
    logging.error("GOOGLE_API_KEY not found...")
    st.stop()
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logging.info("Google Generative AI configured successfully.")
except Exception as e:
    st.error(f"Failed to configure Google AI: {e}") 
    logging.error(f"Failed to configure Google AI: {e}")
    st.stop()


VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash" 
MEMORY_WINDOW_SIZE = 3

USER_ICON = "👤"
ASSISTANT_ICON = "🤖"


@st.cache_resource
def load_base_resources():

    try:
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

        logging.info(f"Checking for vector store at: {VECTOR_STORE_PATH}")
        if not os.path.exists(VECTOR_STORE_PATH):
            st.error(f"Vector store not found at '{VECTOR_STORE_PATH}'. Run kb.py first.")
            logging.error(f"Vector store directory not found: {VECTOR_STORE_PATH}")
            return None, None, None

        logging.info("Loading FAISS vector store...")
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        logging.info("FAISS vector store loaded successfully.")

        logging.info(f"Loading LLM model: {LLM_MODEL}")
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, temperature=0.3, convert_system_message_to_human=True
        )
        logging.info(f"LLM model {LLM_MODEL} loaded successfully.")


        return embeddings, vector_store, llm

    except Exception as e:
        st.error(f"Error loading base resources: {e}") 
        logging.error(f"Error during base resource loading: {e}", exc_info=True)
        return None, None, None


embeddings, vector_store, llm = load_base_resources()


if llm and vector_store:
    if 'memory' not in st.session_state:
        logging.info("Initializing conversation memory in session state.")
        st.session_state.memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE, memory_key="chat_history", return_messages=True, output_key='answer'
        )

    if "messages" not in st.session_state:
        logging.info("Initializing chat message display list in session state.")

        initial_greeting = "Greetings! It is my pleasure to assist you with your insurance policy questions today. How may I be of service?"
        st.session_state.messages = [{"role": "assistant", "content": initial_greeting}]


QA_PROMPT_TEMPLATE = """
You are a courteous, knowledgeable, and humble AI assistant representing our insurance company. Your primary goal is to be exceptionally helpful and polite while answering questions based **strictly** on the information contained within the provided 'Retrieved Documents'.

Instructions:
1.  Carefully review the user's 'Question' and the 'Chat History' to fully understand the context and intent.
2.  Thoroughly search the 'Retrieved Documents' for the relevant information needed to answer the 'Question'.
3.  Compose a clear, helpful, and polite response using **only** information found in the 'Retrieved Documents'. Address the user respectfully.
4.  Begin your response naturally and courteously (e.g., "Certainly, regarding your question...", "I'd be happy to clarify that...", "Based on the policy information I have here..."). Please avoid simply echoing the user's question.
5.  **Humility and Honesty:** If the 'Retrieved Documents' do **not** contain the specific information requested, state this clearly, politely, and humbly. Offer alternative assistance if appropriate. Examples:
    *   "I've carefully reviewed the available documents, but I couldn't locate the specific detail regarding [topic]. My apologies. Perhaps I could assist with another aspect of the policy?"
    *   "While I don't have that specific information in the provided materials, I'd be glad to explain the general process for [related topic], if that would be helpful?"
    *   "That seems like a query best handled by one of our specialists who has access to more detailed account information. Would you like me to guide you on how to contact a human agent?"
    **Crucially, do not invent information or speculate.**
6.  Maintain a consistently helpful, humble, and professional tone throughout the interaction.

Chat History:
{chat_history}

Retrieved Documents:
{context}

Question: {question}

Courteous and Helpful Answer:""" 

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE, input_variables=["chat_history", "context", "question"]
)



conversation_chain = None
if llm and vector_store and 'memory' in st.session_state:
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            return_source_documents=True,
            verbose=False, 
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        logging.info("ConversationalRetrievalChain created successfully with custom persona prompt.")
    except Exception as e:
        st.error(f"Failed to create conversational chain: {e}") 
        logging.error(f"Failed to create conversational chain: {e}", exc_info=True)



def needs_escalation(query: str, response: str) -> bool:

    query_lower = query.lower()
    response_lower = response.lower()
    escalation_keywords = ["human", "agent", "speak to someone", "representative", "talk to person", "escalate", "complex issue", "complaint"]
    if any(keyword in query_lower for keyword in escalation_keywords):
        logging.info(f"Escalation triggered by user keyword: '{query}'")
        return True
    uncertainty_phrases = [
        "couldn't locate the specific detail", "don't have that specific information",
        "not contain the specific information", "unable to find details", "i cannot answer",
        "best handled by one of our specialists", "contacting a human agent", "i don't know",
        "no information found", "not available in the provided materials"
    ]
    if any(phrase in response_lower for phrase in uncertainty_phrases):
         logging.info(f"Escalation triggered by bot uncertainty/suggestion: '{response}'")
         return True
    return False


def stream_response(text: str):
    """Yields characters of the response with a small delay for typing effect."""
    for char in text:
        yield char
        time.sleep(0.02) 
if conversation_chain:
    st.title("🛡️ Insurance Policy Information Assistant")
    st.markdown("Welcome! I am here to assist with your questions regarding our insurance policies. Please ask how I may help you today.")
    st.markdown("---")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
      
            avatar = USER_ICON if message["role"] == "user" else ASSISTANT_ICON
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"]) 


    if user_query := st.chat_input("Please type your question here..."):
        if not conversation_chain:
            st.warning("Apologies, the chat service isn't available at the moment.")
            st.stop()


        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar=USER_ICON):
            st.markdown(user_query)

        with st.spinner("Reviewing the policy information for you..."):
            try:
                logging.info(f"Invoking ConversationalRetrievalChain for query: '{user_query}'")
                result = conversation_chain.invoke({"question": user_query})
                bot_response = result.get('answer', "My apologies, I encountered an issue generating a response.").strip()
                logging.info(f"Generated response: '{bot_response}'")


                is_escalated = needs_escalation(user_query, bot_response)
                escalation_message = "\n\n*Should you require further assistance, especially for complex matters or specific account details not covered in these documents, please do not hesitate to connect with one of our knowledgeable human agents. They would be happy to help.*"
                needs_escalation_text = "contact" in bot_response.lower() or "agent" in bot_response.lower() or "specialist" in bot_response.lower()
                if is_escalated and not needs_escalation_text:
                    bot_response += escalation_message
                    logging.info("Polite escalation message appended.")

            except Exception as e:
                st.error(f"My sincere apologies, an error occurred while processing your request: {e}") 
                logging.error(f"Error during conversational chain invocation: {e}", exc_info=True)
                bot_response = "I seem to have encountered a technical difficulty. Please accept my apologies. You might try again shortly or contact support if the issue persists." 


        st.session_state.messages.append({"role": "assistant", "content": bot_response})


        with st.chat_message("assistant", avatar=ASSISTANT_ICON): 

            st.write_stream(stream_response(bot_response)) 


elif not vector_store or not llm:
     st.warning("Apologies, essential resources failed to load.")
elif not conversation_chain: 
     st.warning("Apologies, chat components failed to initialize.")