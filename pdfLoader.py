import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO

# Interfaz de usuario en Streamlit para ingresar la clave de API
st.title("Análisis de Documentos PDF con LangChain y Ollama")
st.write("Por favor, sube un archivo PDF y luego ingresa tu clave de API para continuar.")

# Interfaz de carga de archivos PDF y clave de API al mismo tiempo
uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")
api_key = st.text_input("Introduce tu API Key de Ollama", type="password")

# Botón de procesar
process_button = st.button("Procesar")

# Continuar solo si ambos, archivo PDF y clave de API están disponibles y el usuario presiona "Procesar"
if uploaded_file is not None and api_key and process_button:
    try:
        # Convertir el archivo PDF cargado en BytesIO
        file_bytes = BytesIO(uploaded_file.read())

        # Cargar el PDF desde el objeto BytesIO
        loader = PyPDFLoader(file_bytes)
        docs = loader.load()

        # Configurar el modelo de chat y embeddings de Ollama
        chatModel = ChatGroq(
            model="llama3-70b-8192",  # Ajusta el nombre del modelo según el que quieras usar
            api_key=api_key
        )
        
        # Usar la API Key para crear OllamaEmbeddings
        ollama_embeddings = OllamaEmbeddings.from_api_key(api_key)  # Asegúrate de que esta sea la forma correcta
        
        # Dividir el texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Crear el índice FAISS
        faiss_index = FAISS.from_documents(splits, ollama_embeddings)
        retriever = faiss_index.as_retriever()

        # Crear la cadena de preguntas y respuestas (QA) con recuperación
        qa_chain = RetrievalQA.from_chain_type(
            llm=chatModel, 
            chain_type="stuff", 
            retriever=retriever, 
            verbose=True
        )

        # Ejemplo de pregunta al modelo
        question = "What is this article about?"
        response = qa_chain.run(question)

        # Mostrar resultados en la interfaz
        st.write("\n---\n")
        st.write("**Pregunta:** ¿De qué trata este artículo?")
        st.write("\n---\n")
        st.write(f"**Respuesta:** {response}")
        st.write("\n---\n")

    except Exception as e:
        st.error(f"Hubo un error: {e}")

elif uploaded_file is None or api_key == "":
    st.warning("Por favor, sube un archivo PDF y proporciona tu clave de API para continuar.")
else:
    st.info("Haz clic en 'Procesar' para comenzar el análisis.")
