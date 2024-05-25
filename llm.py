import config
import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.friendli import ChatFriendli

os.environ["FRIENDLI_TOKEN"] = config.FRIENDLI_TOKEN

llm = ChatFriendli(model="meta-llama-3-70b-instruct")

llm_prompt = ChatOpenAI(
    base_url="https://inference.friendli.ai/dedicated/v1",
    api_key=config.FRIENDLI_TOKEN, 
    model="ougqt9ffuqpb:prompt",
    temperature=0
)

llm_conversation = ChatOpenAI(
    base_url="https://inference.friendli.ai/dedicated/v1",
    api_key=config.FRIENDLI_TOKEN, 
    model="volxoewfla6k:final_conv",
    temperature=0
)