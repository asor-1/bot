import json
import os
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class AdvancedRAG:
    def __init__(self, json_file_path: str, model_name: str = "google/flan-t5-large"):
        self.json_file_path = json_file_path
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="query", output_key="result")

    def load_documents_from_json(self) -> List[Document]:
        with open(self.json_file_path, 'r') as file:
            data = json.load(file)
        
        documents = []
        for item in data:
            content = item.get('content', '')
            metadata = {k: v for k, v in item.items() if k != 'content'}
            documents.append(Document(page_content=content, metadata=metadata))
        
        return self.text_splitter.split_documents(documents)

    def create_vectorstore(self, documents: List[Document]) -> None:
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

    def initialize_llm(self) -> HuggingFacePipeline:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        pipe = pipeline(
            "text2text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        return HuggingFacePipeline(pipeline=pipe)

    def setup_qa_chain(self) -> None:
        llm = self.initialize_llm()
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT,
            },
            memory=self.memory,
        )

    def query(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup() first.")
        
        result = self.qa_chain({"query": question})
        
        answer = result['result']
        sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in result['source_documents']]

        return {
            "answer": answer,
            "sources": sources
        }
    def setup(self) -> None:
        documents = self.load_documents_from_json()
        self.create_vectorstore(documents)
        self.setup_qa_chain()

    def save_vectorstore(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str) -> None:
        self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        self.setup_qa_chain()  # Reinitialize QA chain with loaded vectorstore

    def update_documents(self, new_json_file_path: str) -> None:
        self.json_file_path = new_json_file_path
        new_documents = self.load_documents_from_json()
        self.vectorstore.add_documents(new_documents)
