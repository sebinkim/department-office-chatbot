import streamlit as st

from image import generate_image

import rag

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.write("여러분의 친절한 상담사입니다.")
    url = generate_image(len(st.session_state.messages) // 2)
    st.image(image=url)

prompt = st.chat_input("문의사항을 입력해 주세요.")
if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context, stream = rag.stream_step1(prompt)
        response1 = st.write_stream(stream)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response1
        })
        stream = rag.stream_step2(context, response1)
        response2 = st.write_stream(stream)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response2
        })
else:
    title = st.title(":robot_face: 컴퓨터공학부 학과사무소 챗봇 :robot_face:")
    image = st.image("https://d25fcd02hwfaxf.cloudfront.net/240525/055245908809_fe0711d11a2abd73469450bed683dab6.png")
