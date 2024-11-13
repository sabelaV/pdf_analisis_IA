import streamlit as st
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

# Interfaz de usuario en Streamlit para ingresar la clave de API
st.title("Análisis de Documentos PDF con LangChain y Ollama")
st.write("Por favor, sube un archivo PDF y luego ingresa tu clave de API para continuar.")

# Interfaz de carga de archivos PDF y clave de API al mismo tiempo
uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")
api_key = st.text_input("Introduce tu API Key de Groq", type="password")

# Continuar solo si ambos, archivo PDF y clave de API, están disponibles
if uploaded_file is not None and api_key:
    # Configurar el modelo de chat y embeddings de Ollama
    chatModel = ChatGroq(
        model="llama3-70b-8192",  # Ajusta el nombre del modelo según el que quieras usar
        api_key=api_key
    )
    ollama_embeddings = OllamaEmbeddings(api_key=api_key)
    
    # Cargar el PDF
    loader = PyPDFLoader(uploaded_file)
    docs = loader.load()

    # Dividir el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Generar embeddings para cada fragmento de texto
    document_embeddings = [{"embedding": ollama_embeddings.embed(text["text"])} for text in splits]
    
    # Crear el índice FAISS
    faiss_index = FAISS.from_documents(splits, ollama_embeddings)
    retriever = faiss_index.as_retriever()

    # Definir el sistema y prompt de usuario
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Crear el encadenamiento de preguntas y respuestas
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Ejemplo de pregunta al modelo
    question = "What is this article about?"
    response = rag_chain.invoke({"input": question})

    # Mostrar resultados en la interfaz
    st.write("\n---\n")
    st.write("**Pregunta:** ¿De qué trata este artículo?")
    st.write("\n---\n")
    st.write(f"**Respuesta:** {response['answer']}")
    st.write("\n---\n")

    st.write("**Mostrar metadatos del documento:**")
    st.write(response["context"][0].metadata)

else:
    st.warning("Por favor, sube un archivo PDF y proporciona tu clave de API para continuar.")
