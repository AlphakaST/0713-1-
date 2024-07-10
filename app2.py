import streamlit as st
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

def main():
    st.set_page_config(
        page_title="GPT-4 Chatbot",
        page_icon="🤖"
    )
    st.title("_GPT-4 Chatbot :red[Q&A]_ 🤖")
    st.header("😶주의! 이 챗봇은 참고용으로 사용하세요!", divider='rainbow')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        model_name = st.radio("Select your model", ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo'])
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        
        st.session_state.conversation = get_conversation_chain(openai_api_key, model_name)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요? 😊"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                response = result['answer']
                
                st.markdown(response)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def get_conversation_chain(openai_api_key, model_name):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=None,  # No retriever needed for this simple chatbot
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=False,  # Disable source documents
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
