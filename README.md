AI in Finance – GenAIVersity Hackathon Project
1. Project Title
   AI-Powered Financial Advisor and Risk Analysis Assistant using Generative AI
   
2. Problem Statement
   The finance domain involves complex documents such as bank policies, loan agreements, investment plans, insurance documents, and regulatory guidelines.
   These documents are often lengthy, technical, and difficult for common users to understand.
   
Key challenges:
    Users find it hard to understand financial terms and policies
    Financial advisors are not always available
    Manual query handling is time-consuming and costly
    Incorrect or hallucinated AI responses can cause financial loss
   Therefore, there is a need for a safe, reliable, and document-grounded AI system that can answer financial queries accurately while minimizing risk

3. Proposed Solution
   We propose an AI-powered Financial Assistant built using Generative AI and Retrieval Augmented Generation (RAG).
   The system allows users to ask financial questions in natural language. 
   Instead of answering directly from the language model’s memory, the system retrieves relevant information from verified financial documents and then generates responses using a Large Language Model (LLM).
   
   This approach ensures:
     Accurate and context-based answers
     Reduced hallucinations
     Safer responses for finance-related queries
4. Why Generative AI and RAG in Finance?
   Finance is a high-risk domain where incorrect information can lead to monetary loss. Traditional LLMs may generate confident but incorrect answers.
   
 RAG helps by:
     Grounding responses in verified documents 
     Improving factual accuracy
     Increasing user trust
  Formula:
      LLM + Financial Documents + Retrieval = Reliable Financial AI
      
  5. System Architecture

   The system follows a standard RAG architecture:
        User submits a financial query
        Query is converted into embeddings
        Vector database retrieves relevant document chunks
        Retrieved context is passed to the LLM
        LLM generates a context-aware answer
 Flow:
     User → Embeddings → Vector Database (ChromaDB) → LLM → Answer

6. Detailed Workflow
    Step 1: Document Ingestion
            Financial documents such as loan policies, investment brochures, and FAQs are collected. These documents are cleaned and split into smaller chunks for better retrieval.

   Step 2: Embedding Generation
           Each document chunk is converted into vector embeddings using embedding models. Embeddings capture the semantic meaning of the text.
   
   Step 3: Vector Storage
           The generated embeddings are stored in ChromaDB, which allows fast semantic similarity search.

   Step 4: Query Processing
           When a user asks a question, the query is converted into embeddings and matched against the vector database to retrieve relevant content.

   Step 5: Answer Generation
           The retrieved context along with the user query is sent to the LLM, which generates an answer strictly based on the provided information.

7. Technologies Used

    Programming Language: Python
    Generative AI Framework: LangChain
    Large Language Model: OpenAI / Gemini / Local LLM
    Vector Database: ChromaDB
    Embeddings: OpenAI / HuggingFace
    Version Control: Git and GitHub

8. LLM Parameters and Configuration
    Important LLM parameters used:
   
     Temperature: Low value to reduce randomness
     Max Tokens: Limits response length
     Top-p: Controls output diversity
     Context Window: Defines how much data the model can process
    Low temperature and restricted context are used to make the system safer for financial applications.
   
9. Guardrails and Safety Measures

    Since finance is a sensitive domain, guardrails are applied:
    Responses are restricted to retrieved documents only
    No speculative investment advice is generated
    Warning messages for high-risk financial queries
    Reduced hallucination through RAG grounding

10. Evaluation Strategy

    The system is evaluated based on:
    Relevance of answers
    Accuracy with respect to source documents
    Reduction in hallucinations
    Safety and compliance
    Test queries and manual verification are used for evaluation.

11. Project Structure

    finance-ai-assistant/
│── README.md
│── data/
│ └── financial_documents/
│── ingest.py
│── embeddings.py
│── vector_store.py
│── chatbot.py
│── requirements.txt

11.1 Sample Code Snippets (For Understanding)
Below are simple reference code snippets to demonstrate how the system works. These are minimal examples meant for conceptual clarity and hackathon explanation.
a) Document Ingestion (ingest.py)
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = TextLoader("data/financial_documents/policy.txt")
documents = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
b) Embedding Generation (embeddings.py)
from langchain.embeddings import OpenAIEmbeddings


embeddings = OpenAIEmbeddings()
vector_embeddings = embeddings.embed_documents([chunk.page_content for chunk in chunks])


from langchain.vectorstores import Chroma

c) Vector Store Creation (vector_store.py)
vector_db = Chroma.from_documents(
documents=chunks,
embedding=embeddings,
persist_directory="./chroma_db"
)
vector_db.persist()
d) RAG-based Chatbot (chatbot.py)
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
llm=llm,
retriever=vector_db.as_retriever()
)


query = "What is the interest rate for home loans?"
answer = qa_chain.run(query)
print(answer)                                                                                                                                  

e) Guardrail Example (Prompt Restriction)
system_prompt = """
You are a financial assistant.
Answer only using the provided documents.
If information is not available, say 'Information not found'.
"""

f) requirements.txt
langchain
openai
chromadb
python-dotenv

Note: These code snippets are simplified for learning and hackathon demonstration purposes. Production systems require additional validation and security layers.

finance-ai-assistant/ │── README.md │── data/ │ └── financial_documents/ │── ingest.py │── embeddings.py │── vector_store.py │── chatbot.py │── requirements.txt

## 12. API and Model Usage Notice
- Focus on hallucination reduction and safety
- Designed with future agentic expansion in mind


### 2. Technical Implementation (25%)
- End-to-end pipeline: ingestion → embeddings → vector DB → LLM
- Clean modular Python files
- Reproducible setup using requirements.txt


### 3. Utilization of Artificial Intelligence (25%)
- Use of LLMs for natural language understanding
- Embeddings for semantic similarity
- RetrievalQA chain for grounded generation


### 4. Impact and Expandability (15%)
- Useful for banks, fintech, and users
- Easily extendable to multi-agent systems
- Can integrate compliance and risk agents


### 5. Presentation (10%)
- Clear README documentation
- Simple code snippets
- Easy-to-explain architecture


---


## 17. Reproducible Notebook Example


Below is a **minimal Jupyter Notebook workflow** that reproduces the system logic.


```python
# finance_rag.ipynb (conceptual)
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


# Load documents
loader = TextLoader("data/financial_documents/policy.txt")
docs = loader.load()


# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)


# Create embeddings
embeddings = OpenAIEmbeddings()
vector_db = Chroma.from_documents(chunks, embeddings)


# Build RAG chain
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())


# Query
qa.run("What is the interest rate for home loans?")

18. Minimal FastAPI Service (Optional)


Run command:
  uvicorn app:app --reload

19. Evaluation Notes

 Metrics Used
    Answer relevance
    Context grounding
    Hallucination rate (manual check)
    Safety compliance

 Guardrails
   Restricted prompt
   Low temperature
   Context-only answering

 Limitations
   Dependent on document quality
   Not real-time market data
   Manual evaluation
