import qdrant_client
from langchain.vectorstores import Qdrant
from bs4 import BeautifulSoup
import re
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import BSHTMLLoader
pattern = re.compile(r'<h1>(.*?)</h1>')
# 读取HTML文件
with open("test.html") as file:
    html_content = file.read()

# 使用Beautiful Soup解析HTML
soup = BeautifulSoup(html_content, "html.parser")

# 查找所有的h1标签
h3_tags = soup.find_all('h3')

# 创建一个空列表来存储拆分后的内容
pages = []

law_files = {}
# 遍历所有的h1标签
for i in range(len(h3_tags) - 1):
    # 获取当前h1标签和下一个h1标签之间的内容
    page_content = ""
    current_tag = h3_tags[i]
    title = current_tag.text.strip()
    next_tag = h3_tags[i + 1]
    current_element = current_tag.next_sibling
    while current_element and current_element != next_tag:
        page_content += str(current_element)
        current_element = current_element.next_sibling
        # law_files[title] = html2text.html2text(page_content).replace('#','').replace('*','').strip()
        # law_files[title] = html2text.html2text(page_content).strip()
        law_files[title] = page_content.strip()
print(law_files['中华人民共和国对外关系法'])


    
# html = law_files['中华人民共和国对外关系法']



embeddings = HuggingFaceEmbeddings(
            model_name='./GanymedeNil_text2vec-large-chinese',
            model_kwargs={'device': 'cuda'}
        )
client = qdrant_client.QdrantClient(
    url="http://localhost:23112", prefer_grpc=False
)


for title, contents in law_files.items():    
    # with open(f'laws/{title}.html', 'w') as file:
    #     file.write(contents)

    soup = BeautifulSoup(contents, 'html.parser')
    li_contents = [li_tag.text for li_tag in soup.find_all('li')]
    qdrant = Qdrant.from_texts(
        li_contents, 
        embedding=embeddings,
        url="http://localhost:23112",
        collection_name="labor_law"
    )


os.chdir('/root/test_search/test_law')
html_files = os.listdir('laws')

# for f in html_files:
#     loader = BSHTMLLoader(f"laws/{f}")
#     data = loader.load()
#     print(data[0].page_content)

