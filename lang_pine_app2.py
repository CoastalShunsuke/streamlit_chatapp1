import streamlit as st
import os
from dotenv import load_dotenv
import pinecone  # ここを修正
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# ...（省略）...

# Pinecone の初期化
if pinecone_api_key:
    try:
        # Pineconeの初期化を追加
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
if "history" not in st.session_state:
    st.session_state["history"] = []

# ユーザーの質問入力ボックス
st.subheader("質問を入力してください：")
user_input = st.text_input("質問", placeholder="例: 最近のAI技術のトレンドは？")

if user_input and openai_api_key and pinecone_api_key:
    # OpenAI Embeddingsの初期化
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # **Pineconeのインデックスを読み込む前に初期化が必要**
    # Pineconeのインデックスを読み込む
    vector_store = LangChainPinecone.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    # OpenAI LLM の初期化
    llm = OpenAI(openai_api_key=openai_api_key, temperature=temperature)

    # これまでの会話履歴をまとめたプロンプトを作成
    conversation_history = "\n".join(
        [f"ユーザー: {user}\nアシスタント: {assistant}" for user, assistant in st.session_state["history"]]
    )

    # 現在の質問を履歴に追加してプロンプトを生成
    if conversation_history:
        prompt = f"{conversation_history}\nユーザー: {user_input}\nアシスタント:"
    else:
        prompt = f"ユーザー: {user_input}\nアシスタント:"

    # RetrievalQA の作成
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True
    )

    # 質問に対する回答と参照された文書を取得
    result = qa_chain({"query": prompt})

    response = result['result']
    source_documents = result['source_documents']

    # チャット履歴に質問と回答を追加
    st.session_state["history"].append((user_input, response))

    # チャット履歴の表示
    with st.expander("💬 チャット履歴", expanded=True):
        for i, (user, assistant) in enumerate(st.session_state["history"]):
            st.markdown(f"**ユーザー ({i+1})**: {user}")
            st.markdown(f"**アシスタント**: {assistant}")

    # 参照された文書の表示
    st.subheader("🔍 参照された文書の一部:")
    for doc in source_documents:
        st.markdown(f"**文書内容**: {doc.page_content}")
