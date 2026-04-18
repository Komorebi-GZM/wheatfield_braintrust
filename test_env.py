import streamlit
import langchain
import paddleocr
import chromadb
import sentence_transformers

print("Streamlit version:", streamlit.__version__)
print("LangChain version:", langchain.__version__)
print("PaddleOCR version:", paddleocr.__version__)
print("ChromaDB version:", chromadb.__version__)
print("Sentence Transformers version:", sentence_transformers.__version__)
print("All dependencies are installed successfully!")
