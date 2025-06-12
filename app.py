
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="GPT Jurídico da Dê", layout="wide")
st.title("⚖️ GPT Jurídico da Dê")
st.markdown("Cole ou envie a sentença e razões de apelação. O sistema consulta suas decisões e entrega um voto estruturado no padrão da 29ª Câmara.")

api_key = st.text_input("🔑 Cole sua OpenAI API Key:", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    # Carregar base vetorial
    with st.spinner("🔍 Carregando jurisprudência da Dê..."):
        loader = DirectoryLoader("minutas_anonimizadas", glob="**/*.docx")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = splitter.split_documents(docs)
        db = Chroma.from_documents(documents, OpenAIEmbeddings())
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0),
            chain_type="stuff",
            retriever=db.as_retriever()
        )

    st.success("Jurisprudência carregada!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("📄 **Sentença**")
        sentenca = st.text_area("Cole aqui ou faça upload ao lado", height=200, key="sentenca")
        sent_file = st.file_uploader("Ou envie .docx", type=["docx"], key="sent_file")

    with col2:
        st.markdown("📄 **Razões de apelação**")
        apela = st.text_area("Cole aqui ou faça upload ao lado", height=200, key="apela")
        apela_file = st.file_uploader("Ou envie .docx", type=["docx"], key="apela_file")

    def read_docx(file):
        from docx import Document
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if sent_file:
        sentenca = read_docx(sent_file)
    if apela_file:
        apela = read_docx(apela_file)

    if st.button("📝 Gerar voto"):
        with st.spinner("Gerando minuta com base nas decisões anteriores..."):
            prompt = f"""
Você é assistente jurídico da 29ª Câmara de Direito Privado do TJSP.
Redija um voto de apelação com base na seguinte sentença e razões recursais:

Sentença: {sentenca}

Apelação: {apela}

Use estrutura técnica, linguagem objetiva, e fundamente conforme jurisprudência indexada.
"""
            resposta = qa.invoke({"query": prompt})
            st.markdown("### ✨ Voto gerado:")
            st.text_area("Resultado", value=resposta["result"], height=400)
