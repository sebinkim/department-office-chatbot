import streamlit as st

import rag

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("문의사항을 입력해 주세요.")
if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # stream = rag.stream(st.session_state.messages)
        # response = st.write_stream(stream)
        response = rag.invoke(st.session_state.messages)
        st.write(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
else:
    title = st.title(":robot_face: 컴퓨터공학부 학과사무소 챗봇 :robot_face:")
