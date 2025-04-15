# from langchain.chains import ConversationalRetrievalChain
# from langchain_openai import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
#
# def get_conversational_chain():
#     # Load vector store
#     db = Chroma(
#         persist_directory="vectorstore",
#         embedding_function=OpenAIEmbeddings()
#     )
#
#     # Conversation memory
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )
#
#     # LLM
#     llm = ChatOpenAI(temperature=0)
#
#     # Combine all into a chain
#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=db.as_retriever(),
#         memory=memory
#     )
#
#     return chain


from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma

def get_conversational_chain():
    db = Chroma(
        persist_directory="vectorstore",
        embedding_function=OpenAIEmbeddings()
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # ✅ THIS is crucial
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=db.as_retriever(),
        memory=memory,
        return_source_documents=True  # ✅ Keep this to display sources
    )

    return chain
