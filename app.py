import streamlit as st
from fastembed import TextEmbedding
import chromadb
from openai import OpenAI

@st.cache_resource
def load():
    m = TextEmbedding("BAAI/bge-small-zh-v1.5", cache_dir="C:/Users/Lenovo/CherryModels")
    return m

@st.cache_resource
def load_col():
    kb = chromadb.PersistentClient(path="kb")
    return kb.get_or_create_collection(name="doc60-20")
                                                                               
@st.cache_resource
def load_client():
    f = open("key.txt", "r", encoding="UTF-8")
    key = f.read().strip()
    f.close()
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")

m = load()
col = load_col()
client = load_client()
up = st.file_uploader("上传一个 txt 文件", type="txt")
if up:
    zhi = up.read().decode("utf-8")
    chunks = []
    n = 60
    overlap = 20

    for i in range(0, len(zhi), n - overlap):
        chunks.append(zhi[i:i+n])
    vecs = list(m.embed(chunks))
    ids = []

    for i in range(len(chunks)):
        ids.append(f"{up.name}_{i}")
    col.add(documents=chunks, ids=ids, embeddings=vecs)
    st.success(f"已导入 {len(chunks)} 个片段")
    
def ask(wen):
    w = list(m.embed([wen]))
    r = col.query(query_embeddings=w, n_results=3)
    zl = ""
    for d in r["documents"][0]:
        zl = zl + d + "\n\n"
    p = f"""请只根据下面的资料回答问题。
如果资料里没有答案，就直接回答"资料中没有提到"，不要编。
问我刚才聊了什么这类可以说，可以人性化一点

资料：
{zl}

问题：{wen}"""
    lishi = []
    for x in st.session_state["hua"]:
        lishi.append({"role": x["role"], "content": x["content"]})
    lishi.append({"role": "user", "content": p})
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=lishi
    )
    return resp.choices[0].message.content

if "hua" not in st.session_state:
    st.session_state["hua"] = []

q = st.chat_input("问点什么")
if q:
    da = ask(q)
    st.session_state["hua"].append({"role": "user", "content": q})
    st.session_state["hua"].append({"role": "assistant", "content": da})

for x in st.session_state["hua"]:
    st.chat_message(x["role"]).write(x["content"])