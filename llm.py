import config
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://inference.friendli.ai/dedicated/v1",
    api_key=config.FRIENDLI_TOKEN, 
    model="v54gk3uli2eg:conv",
    temperature=0
)