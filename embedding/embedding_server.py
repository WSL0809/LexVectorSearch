from langchain.embeddings import HuggingFaceEmbeddings
from fastapi import FastAPI

import uvicorn
# import os
# os.environ['HTTP_PROXY'] = 'http://proxy_url:proxy_port'
# os.environ['HTTPS_PROXY'] = 'http://proxy_url:proxy_port'
app = FastAPI()
embeddings = HuggingFaceEmbeddings(
model_name='GanymedeNil_text2vec-large-chinese',
model_kwargs={'device': 'cuda'}
)



@app.get('/embeddings/{sentance}')
async def embedding_server(sentance: str):
    return embeddings.embed_query(sentance) 
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=23111)
