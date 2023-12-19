from fastapi import FastAPI, HTTPException
from typing import List
import requests
import os
import uvicorn
import qdrant_client
import json
import qianfan
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from fastapi.middleware.cors import CORSMiddleware
API_KEY = "8bE4QaIOMIHZDB5ZQcwn6ZgK"
SECRET_KEY = "TIH3i5d0zEIfRO3Sq6LLnF7QrrWxuYKh"

app = FastAPI()
embeddings = HuggingFaceEmbeddings(
            model_name='./GanymedeNil_text2vec-large-chinese',
            model_kwargs={'device': 'cuda'}
        )
origins = [
    "https://fa.wiki.xin",
    "https://wiki.xin",
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8081",
    "http://10.10.10.20:4001",
    "http://search.wiki.xin",
    "https://search.wiki.xin"
]

# 3、配置 CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许访问的源
    allow_credentials=True,  # 支持 cookie
    allow_methods=["*"],  # 允许使用的请求方法
    allow_headers=["*"]  # 允许携带的 Headers
)

def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def chatlaw(message, laws):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    print('loading LLM Model')
    payload = json.dumps({
    "messages": [
        {
            "role": "user",
            "content": f"你现在是一个法律条文解读师。我将描述一种法律情况，并且提供相关的法律法规给你参考，您根据法律情况，解把我提供给你的法律法规解释清楚即可。法律情况：{message}。可供参考的法律法规：{laws}"
        }
    ],
    "top_p": 0.2,
    "extra_parameters": {
        "use_keyword": True,
        "use_reference": True
    }
})
    headers = {
        'Content-Type': 'application/json'
    }
    print(f'sending request {payload}')
    response = requests.request("POST", url, headers=headers, data=payload)
    print(f'response: {response.text}')
    return response
def search(q):
    try:
        client = qdrant_client.QdrantClient(
            url="http://localhost:23112", prefer_grpc=False
        )
        qdrant = Qdrant(
            client=client, collection_name="labor_law",
            embeddings=embeddings
        )
        res = []
        found_docs = qdrant.max_marginal_relevance_search(q)
        for i in found_docs:
            res.append(i.page_content)
    except Exception as e:
        print(f"An error occurred: {e}")
        # 可能还想记录更详细的错误信息或者根据错误类型做不同处理
        raise HTTPException(status_code=500, detail="Internal server error")
    return res


@app.get('/query/law/{q_content}')
def hello_world(q_content: str):  # FastAPI function
    try:
        print(q_content)
        qdrant_res = search(q_content)
        print(qdrant_res)
        json_str = json.dumps(qdrant_res)
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 如果结果不能被序列化为 JSON，返回一个错误
        raise HTTPException(status_code=500, detail="Data serialization error")
    except HTTPException as http_ex:
        # 这里可以根据不同的HTTP异常做不同的处理
        raise http_ex
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get('/query/chat/{q_content}')
def hello_world(q_content: str):  # FastAPI function
    try:
        print(q_content)
        qdrant_res = search(q_content)
        print(qdrant_res)
        json_str = json.dumps(qdrant_res)
        # res = chatlaw(q_content, json.loads(json_str))
        res = chatlaw(q_content, qdrant_res)
        return res.json()
    except json.JSONDecodeError:
        # 如果结果不能被序列化为 JSON，返回一个错误
        raise HTTPException(status_code=500, detail="Data serialization error")
    except HTTPException as http_ex:
        # 这里可以根据不同的HTTP异常做不同的处理
        raise http_ex
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 主函数
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=23111)
