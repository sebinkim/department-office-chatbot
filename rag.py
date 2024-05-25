import requests
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder

import config
from llm import llm, llm_prompt, llm_conversation

# document_ids = ["OpSYpHpaiDEL", "REn1dvZbezcE", "qJ0OeAqYUajP", "Uufw2nnFQzxK"]
document_ids = ["PVDxtI4DS65R", "GrMXVtV7ae9I", ] #"17t5nUTgoypK", "0CVkcIbw95ml", "I7EP4FXwXEDl"]

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
대화가 주어지면, 아래 제공된 context를 기반으로 질문에 답변하세요.
당신은 "서울대학교 컴퓨터공학부 학과사무실"의 챗봇입니다. 서울대학교 컴퓨터공학부와 관련된 질문만 답변하십시오.

만약 관련 없는 질문이 들어오면, 다음과 같이 답변하십시오: "저는 서울대학교 컴퓨터공학부 학과사무실 챗봇입니다. 컴퓨터공학부와 관련된 정보만 질문해 주세요."

만약 필요한 정보를 찾지 못하거나 답변이 불확실하면, 다음과 같이 안내하십시오: "해당 정보를 찾지 못했습니다. 추가적인 도움이 필요하시면, 문의하고자 하는 내용과 함께 연락처를 남겨 주시면 관련 교직원의 연락처를 안내해 드리겠습니다. 또는 학과 홈페이지(https://cse.snu.ac.kr/)에서 직접 정보를 찾아보실 수도 있습니다."

답변의 길이는 1000자를 넘지 마시오.

질문에 대한 답변을 생성할 때는 정확한 한국어 맞춤법을 사용하고, 정중하게 작성하십시오.

<context>
{context}
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
question_answering_chain = question_answering_prompt | llm_conversation

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
위 대화만을 참고해, 사용자가 필요한 정보를 정확히 얻기 위한 질문을 한국어로 생성하세요.
질문 외에 다른 문장은 만들지 마세요."
"""

"""
RAG를 이용해 사용자가 요청한 정보를 답하기 위한 검색 query를 생성하세요.
생성된 query는 사용자의 질문에 대한 답변을 찾는 데 필요한 모든 관련 정보를 포함해야 합니다. 
쿼리 외에 다른 문장이나 설명은 작성하지 마세요.
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

CONTEXT_REFINEMENT_TEMPLATE = """
아래 주어진 'context'에서 'answer'와 직접적으로 관련 없는 정보를 제거하시오.
raw data 그대로 발췌하여 제공해야 하며, 어떠한 수정이나 요약도 하지 마시오.
관련 정보는 하나의 연속된 덩어리로 뽑아내고, 응답의 맨 처음에 "**발췌한 텍스트의 원문은 아래와 같습니다:**"를 넣으시오.

<context>
{context}

<answer>
{answer}
"""


context_refinement_prompt = PromptTemplate.from_template(CONTEXT_REFINEMENT_TEMPLATE)

context_refinement_chain = context_refinement_prompt | llm_prompt

# Run

# def stream(history: list[dict]):
#     messages = [
#         HumanMessage(content=mespromptsage["content"]) if message["role"] == "user" else \
#         AIMessage(content=message["content"]) \
#             for message in history
#     ]
#     return conversational_retrieval_chain.stream(
#         {
#             "messages": messages
#         }
#     )

def stream_step1(prompt: str):
    messages = [
        HumanMessage(content=prompt)
    ]

    query = query_transforming_retrieval_chain.invoke({
        "messages": messages
    })

    print("Query Transform Prompt:", query_transform_prompt.invoke({
        "messages": messages
    }))
    print("Generated Query:", query)
    print()

    context = custom_retriever(query)

    print("Retrieved Context:", context)
    print()

    print("Question Answering Prompt:", question_answering_prompt.invoke({
        "messages": messages,
        "context": context,
    }))

    stream = question_answering_chain.stream({
        "messages": messages,
        "context": context,
    })

    return context, stream

def stream_step2(context, answer):
    return context_refinement_chain.stream({
        "context": context,
        "answer": answer
    })