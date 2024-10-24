import streamlit as st
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# アプリのタイトルと説明
st.set_page_config(page_title="Conversational RAG Chatbot", page_icon="💬")
st.title("💬 Conversational RAG Chatbot")
st.markdown("""
このアプリは、会話履歴を考慮した Retrieval-Augmented Generation (RAG) チャットボットです。
Pinecone のベクトルストアと OpenAI の GPT を組み合わせ、過去の履歴を踏まえて対話を行います。
""")

# サイドバーの設定
st.sidebar.title("🛠 パラメータ調整")
temperature = st.sidebar.slider("Temperature (生成の多様性)", 0.0, 1.0, 0.7, step=0.1)
top_k = st.sidebar.slider("Top-k Documents (検索時の上位文書数)", 1, 10, 3)

# 環境変数の取得（st.secrets を使用）
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENVIRONMENT"]
index_name = st.secrets["PINECONE_INDEX_NAME"]

# Pinecone の初期化
if pinecone_api_key:
    try:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
            )
            st.success(f"新しいインデックス '{index_name}' が作成されました。")
        else:
            st.success(f"インデックス '{index_name}' が正常にロードされました。")
    except Exception as e:
        st.error(f"Pinecone 初期化エラー: {e}")
else:
    st.error("Pinecone API Key が指定されていません。")

# セッションステートの初期化
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ユーザーの質問入力ボックス
st.subheader("質問を入力してください：")
user_input = st.text_input("質問", placeholder="例: 最近のAI技術のトレンドは？")

if user_input and openai_api_key and pinecone_api_key:
    # OpenAI Embeddings の初期化
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Pinecone のインデックスを読み込む
    vector_store = LangChainPinecone.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    # OpenAI Chat Model の初期化
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature)

    # Conversational Retrieval Chain の作成
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True
    )

    # 会話履歴を取得
    chat_history = st.session_state["chat_history"]

    # 質問に対する回答と参照された文書を取得
    result = qa_chain({"question": user_input, "chat_history": chat_history})

    response = result["answer"]
    source_documents = result["source_documents"]

    # チャット履歴に質問と回答を追加
    chat_history.append((user_input, response))
    st.session_state["chat_history"] = chat_history

    # チャット履歴の表示
    with st.expander("💬 チャット履歴", expanded=True):
        for i, (user, assistant) in enumerate(chat_history):
            st.markdown(f"**ユーザー ({i+1})**: {user}")
            st.markdown(f"**アシスタント**: {assistant}")

    # 参照された文書の表示
    st.subheader("🔍 参照された文書の一部:")
    for doc in source_documents:
        st.markdown(f"**文書内容**: {doc.page_content}")
