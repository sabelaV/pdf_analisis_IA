import streamlit as st
from langchain_groq import ChatGroq

# Interfaz de usuario en Streamlit para ingresar la clave de API de Groq
st.title("Análisis de Documentos PDF con LangChain y Groq")
st.write("Por favor, ingresa tu clave de API de Groq para continuar.")

# Pedir clave de API de Groq directamente en la interfaz de Streamlit
groq_api_key = st.text_input("Introduce tu Groq API Key", type="password")

# Continuar solo si la clave de Groq está disponible
if groq_api_key:



chatModel = ChatGroq(

    model="llama3-70b-8192"  # Ajusta el nombre del modelo según el que quieras usar

)


# Interfaz de carga de archivos PDF
uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")
if uploaded_file is not None:
    # Cargar el PDF
    loader = PyPDFLoader(uploaded_file)
    docs = loader.load()

    # Dividir el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Crear el vectorstore y el recuperador
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

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
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
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
    st.warning("Por favor, introduce tu clave de API de Groq.")
