from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = Docx2txtLoader("example.docx")
documents = loader.load()

# 查看加载结果
print(f"加载了 {len(documents)} 个文档片段")
print("----------------------------------")
print(f"第一段内容预览: {documents[0].page_content[:200]}...")
print("----------------------------------")
print(f"元数据: {documents[0].metadata}")

# 创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个文本块的最大字符数
    chunk_overlap=100,  # 相邻文本块之间的重叠字符数（保持上下文连贯性）
    separators=["\n\n", "\n", "。", "！", "？", "，", ""],  # 定义了分割符的优先级顺序
    add_start_index=True,  # 记录每个块在原文档中的起始位置
)

# 执行分割
all_splits = text_splitter.split_documents(documents)
print(f"分割后得到 {len(all_splits)} 个文本块")
print(all_splits[0].page_content)

from dotenv import load_dotenv
import os
load_dotenv()

ollama_embedding_model = "bge-m3"
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model=ollama_embedding_model)

query = "怎么报销？"
query_embedding = embeddings.embed_query(query)
print(f"查询向量维度: {len(query_embedding)}")
print(f"查询向量前5个值: {query_embedding[:5]}")

from langchain_community.vectorstores import FAISS

#vectorstore = FAISS.from_documents(documents, embeddings)
# 保存向量数据库到本地
#vectorstore.save_local("faiss_vectorstore_docx")

import numpy as np
#NumPy 库中的线性代数模块
import numpy.linalg as norm
question = "怎么申请离职？"
question_embedding = embeddings.embed_query(question)

def cos_sim(a,b):
    """计算余弦相似度，数值越大越相似"""
    return np.dot(a, b) / (norm.norm(a) * norm.norm(b))

def l2_dist(a,b):
    """计算欧氏距离，数值越小越相似"""
    return norm.norm(np.asarray(a) - np.asarray(b))

# 计算相似度
print(f"余弦相似度: {cos_sim(question_embedding, query_embedding)}")
print(f"欧氏距离: {l2_dist(question_embedding, query_embedding)}")

# 查询向量数据库
vectorstore = FAISS.load_local(
    "faiss_vectorstore_docx",
    embeddings,
    allow_dangerous_deserialization=True
)

docs_with_score = vectorstore.similarity_search_with_score(question, k = 3) # 返回 top 3 最相似的文档
for i, (doc, score) in enumerate(docs_with_score):
    print(f"----- 第 {i+1} 个相似文档 -----")
    print(f"相似度分数: {score}")
    print(f"内容预览: {doc.page_content[:200]}...")
    print("----------------------------------")

context = "\n".join([doc.page_content for doc, score in docs_with_score])

print("----- 拼接后的上下文 -----")
prompt = f"""根据以下内容，回答用户的问题。如果无法从内容中得到答案，请说“抱歉，我无法回答这个问题”。
内容: {context}
用户问题: {question}
答案:"""
print(prompt)

from langchain_ollama import ChatOllama
ollama_chat_model = "qwen2.5:7b"
chat = ChatOllama(model=ollama_chat_model)
response = chat.invoke(prompt)
print("----- 模型回答 -----")
print(response)
