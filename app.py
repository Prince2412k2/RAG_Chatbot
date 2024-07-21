import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from main import Chat

model=Chat()
if "chat_history" not in st.session_state:
  st.session_state.chat_history=[]

st.set_page_config(page_title="Silverbot")
# st.image("assets/eve.png", width=100)
st.header("Silverbot")

with st.chat_message("👾"):
  st.markdown("How can i help you?")

#conversation
for message in st.session_state.chat_history:
  if isinstance(message,HumanMessage):
    with st.chat_message("🗣️"):
      st.markdown(message.content)
  else:
    with st.chat_message("👾"):
      st.markdown(message.content)



query=st.chat_input("Your Query ")

if query is not None and query!="":

  st.session_state.chat_history.append(HumanMessage(query))
  with st.chat_message("🗣️"):
    st.markdown(query)

  with st.chat_message("👾"):
    output=model.invoke(query)
    rem=st.write_stream(output)
    # st.markdown()

  st.session_state.chat_history.append(AIMessage(rem))
