import requests
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models.friendli import ChatFriendli

import config
from llm import llm

os.environ["FRIENDLI_TOKEN"] = config.FRIENDLI_TOKEN

# document_ids = ["OpSYpHpaiDEL", "REn1dvZbezcE", "qJ0OeAqYUajP", "Uufw2nnFQzxK"]
document_ids = ["GrMXVtV7ae9I", "7xTiDeHuuEAD", "17t5nUTgoypK", "0CVkcIbw95ml"]

def retrieve_contexts(document_ids: list[str], query: str, k: int) -> list[str]:
    resp = requests.post(
        "https://suite.friendli.ai/api/beta/retrieve",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.FRIENDLI_TOKEN}",
        },
        json={
            "document_ids": document_ids,
            "query": query,
            "k": k,
        }
    )
    data = resp.json()
    return [r["content"] for r in data["results"]]

# https://python.langchain.com/v0.1/docs/use_cases/chatbots/

# Step 1. Question answering chain

SYSTEM_TEMPLATE = """
대화가 주어지면, 아래 context를 기반으로 질문에 답하세요.
질문에 답을 알지 못하면 답을 지어내지 말고 "죄송합니다. 더 자세한 내용은 (02) 880-7288로 문의해 주세요."라고 답하세요.
한국어로 맞춤법을 지켜 정중하게 답하세요.

<context>
{context}
</context>
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
question_answering_chain = question_answering_prompt | llm

# Step 2. Retrieval chain

# def custom_retriever(params: dict): 
#     results = retrieve_contexts(document_ids, question, 3)
#     return "\n\n".join(results)

def custom_retriever(question: str): 
    results = retrieve_contexts(document_ids, question, 3)
    return "\n\n".join(results)

# retrieval_chain = RunnablePassthrough \
#     .assign(context=custom_retriever) \
#     .assign(answer=document_chain)

# stream = retrieval_chain.stream(
#     {
#         "messages": [
#             HumanMessage(content="컴퓨터공학부 졸업 규정이 뭐야"),
#             AIMessage(content="""'컴퓨터공학부 졸업 규정은 입학년도에 따라 다르며, 졸업 요건은 다음과 같습니다.\n\n* 2011~2014학번: 63학점 이수(전필 36학점 + 전선 내규 5학점을 포함)\n* 2008~2010학번: 60학점 이수(전필 33학점 + 전선 내규 5학점을 포함)\n* 2025학번: 63학점 이수(전필 27학점 + 전선 내규필수 5학점을 포함)\n* 2021~2024학번: 63학점 이수(전필 30학점 + 전선 내규필수 8학점을 포함)\n* 2020학번: 63학점 이수(전필 31학점 + 전선 내규필수 8학점을 포함)\n* 2019학번: 63학점 이수(전필 35학점 + 전선 내규필수 4학점을 포함)\n* 2015~2018학번: 63학점 이수(전필 37학점 + 전선 내규 4학점을 포함)\n\n이 외에도 각 학번마다 전필, 전선 내규필수 과목이 다르므로, 자세한 사항은 해당 학번의 졸업 규정을 참조해야 합니다.'"""),
#             HumanMessage(content="방금 했던 말 반복해봐")
#         ]
#     }
# )

# Step 3. Query transform chain

QUERY_TRANSFORM_TEMPLATE = """
위 대화만을 참고해, 데이터베이스에서 필요한 정보를 얻기 위한 검색 쿼리를 생성하세요.
쿼리 외에 다른 문장은 만들지 마세요."
"""

query_transform_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        ("user", QUERY_TRANSFORM_TEMPLATE),
    ]
)

query_transforming_retrieval_chain = \
    query_transform_prompt | llm | StrOutputParser()


# Step 4. Context refinement chain

CONTEXT_REFINEMENT_PROMPT = """
아래 주어진 context에서 아래 주어진 query과 관련 있는 내용만 남기고 나머지는 제거해 답변하세요.
필요한 내용만 답변하고, 다른 문잗은 추가하지 마세요.

<context>
{context}
</context>
<query>
{query}
</query>
"""


context_refinement_prompt = PromptTemplate.from_template(CONTEXT_REFINEMENT_PROMPT)

context_refinement_chain = context_refinement_prompt | llm

# Run

# def stream(history: list[dict]):
#     messages = [
#         HumanMessage(content=message["content"]) if message["role"] == "user" else \
#         AIMessage(content=message["content"]) \
#             for message in history
#     ]
#     return conversational_retrieval_chain.stream(
#         {
#             "messages": messages
#         }
#     )

def invoke(history: list[dict]):
    messages = [
        HumanMessage(content=message["content"]) if message["role"] == "user" else \
        AIMessage(content=message["content"]) \
            for message in history
    ]

    query = query_transforming_retrieval_chain.invoke({
        "messages": messages
    })
    context = custom_retriever(query)
    answer = question_answering_chain.invoke({
        "messages": messages,
        "context": context,
    }).content

    truncated_context = context_refinement_chain.invoke({
        "context": context,
        "query": query
    }).content

    return str(answer) + "\n\n[참고 자료]\n\n" + str(truncated_context)