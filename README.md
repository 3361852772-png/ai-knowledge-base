# AI 知识库

基于 RAG 的文档问答工具。上传 txt 文档，用中文提问，AI 只根据文档内容回答；
资料里没有的会说"资料中没有提到"，不编造。

## 功能
- 上传 txt 文档，自动切块
- 自动向量化并存入本地向量库
- 基于文档内容回答问题，不瞎编

## 技术栈
Python / Streamlit / ChromaDB / fastembed / DeepSeek API

## 怎么运行
1. `pip install -r requirements.txt`
2. 在 `key.txt` 里填入你的 DeepSeek API Key
3. `streamlit run app.py`
