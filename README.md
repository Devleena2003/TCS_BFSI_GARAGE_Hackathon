# 🛡️ AI-Powered Insurance Policy Information Chatbot

## 📌 Problem Statement

Insurance companies provide various policies such as health, life, auto, and home insurance. Customers often have questions about these policies, including coverage, premiums, and claims. Providing timely, accurate information is crucial. This project aims to build an **AI-powered chatbot** to assist customers with such queries using natural language understanding and a document-based knowledge base.

---

## 🎯 Objective

Develop a chatbot that:
- Answers natural language queries about insurance policies.
- Retrieves accurate information from a PDF-based knowledge base.
- Escalates complex issues to human agents.
- Provides a user-friendly interface.

---

## 🧠 Methodology

### 1. **Knowledge Base Creation**
- **Source:** SBI online insurance policy PDF.
- **Loader:** `PyPDFDirectoryLoader` loads documents from the `data/` folder.
- **Text Splitting:** `RecursiveCharacterTextSplitter` splits documents into chunks of 1000 characters with 150 overlap.
- **Embeddings:** Generated using `GoogleGenerativeAIEmbeddings` (`embedding-001`).
- **Vector Store:** FAISS is used to store and retrieve document chunks.

### 2. **Chatbot Interface**
- **UI Framework:** Built using `Streamlit`.
- **LLM Integration:** `gemini-1.5-flash` is used for generating natural language responses.
- **Retrieval Chain:** Implemented using `ConversationalRetrievalChain` with a custom prompt.
- **Memory Management:** Maintained with `ConversationBufferWindowMemory`.

### 3. **Prompt Design**
- Persona-based prompt ensures responses are courteous, accurate, and document-grounded.
- Politely acknowledges when information is unavailable and suggests escalation.

### 4. **Escalation Mechanism**
- Recognizes phrases indicating need for human support.
- Appends a polite escalation message when necessary.

---

## ✅ Results

- Successfully loads policy PDF and creates a searchable knowledge base.
- Accurately retrieves and answers user queries via natural conversation.
- Escalates issues appropriately when information is unavailable.
- Offers a clean and responsive chatbot interface.

---

## 📌 Conclusion

The chatbot efficiently handles insurance-related queries using LLM and a PDF-based knowledge base. It improves user experience by delivering clear, reliable information and routing complex queries to human agents when needed.

---

## 💡 Why This Approach?

- **Google Generative AI:** Powerful for both embeddings and natural language generation.
- **FAISS:** Efficient for document retrieval and semantic similarity search.
- **Streamlit:** Rapid UI development for demos and prototyping.
- **LangChain:** Simplifies chaining LLMs with retrieval, memory, and prompts.

---

## 🚀 How to Run

1. Add insurance PDFs to the `data/` folder.
2. Run `kb.py` to create the vector store.
3. Launch the chatbot:  
   ```bash
   streamlit run app.py
