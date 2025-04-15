import streamlit as st
from dotenv import load_dotenv
import os
from datetime import datetime
from scripts.chat_chain import get_conversational_chain
from scripts.github_ticket import create_github_issue

# Load secrets
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = "im-fr0stman/customer-support-chatbot.0"

# Page setup
st.set_page_config(page_title="Customer Support Chatbot", layout="wide")
st.title("🤖 Customer Support Assistant")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chain = get_conversational_chain()

# Input form
with st.form(key="chat_form"):
    user_question = st.text_input("You:", placeholder="Type your message...")
    submit_button = st.form_submit_button("Send")

if submit_button and user_question:
    # Chat logic
    response = st.session_state.chain.invoke({"question": user_question})
    bot_reply = response["answer"]

    # Add sources
    sources = response.get("source_documents", [])
    source_info = ""
    for doc in sources:
        metadata = doc.metadata
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        source_info += f"- 📄 **{source}**, page {page}\n"

    if source_info:
        # Wrap sources inside a Streamlit expander
        bot_reply += "\n\n<details><summary><strong>📄 Sources</strong></summary>\n\n" + source_info + "\n\n</details>"

    # Detect if response was unhelpful
    unhelpful_phrases = [
        "i don't know", "i'm not sure", "no relevant information",
        "cannot find", "sorry, i don't have", "i don't have information",
        "i don't have enough information", "i have no data", "i couldn't find"
    ]
    if any(phrase in bot_reply.lower() for phrase in unhelpful_phrases):
        bot_reply += "\n\n🚨 *It looks like I couldn't answer your question. Would you like to create a support ticket?*"

    # Add timestamp and save
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.chat_history.append(("You", user_question, now))
    st.session_state.chat_history.append(("Bot", bot_reply, now))

# Display chat history
for role, text, time in st.session_state.chat_history:
    if role == "You":
        st.markdown(
            f"""
            <div style='text-align:right; margin-bottom: 10px;'>
                <div style='display: inline-block; background-color: #dcf8c6; padding: 8px 12px; border-radius: 10px; max-width: 80%;'>
                    <span style='font-size: 16px;'>{text}</span><br>
                    <small>{time} 👨‍💻</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='text-align:left; margin-bottom: 10px;'>
                <div style='display: inline-block; background-color: #f1f0f0; padding: 8px 12px; border-radius: 10px; max-width: 80%;'>
                    <span style='font-size: 16px;'>{text}</span><br>
                    <small>🤖 {time}</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Show support ticket form if bot couldn't answer
if any("support ticket" in msg[1] for msg in st.session_state.chat_history if msg[0] == "Bot"):
    with st.expander("📝 Create Support Ticket"):
        with st.form("ticket_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            title = st.text_input("Ticket Title", value=user_question)
            description = st.text_area("Description")

            submitted = st.form_submit_button("Submit Ticket")
            if submitted:
                success, result = create_github_issue(
                    name=name,
                    email=email,
                    title=title,
                    description=description,
                    repo=GITHUB_REPO,
                    token=GITHUB_TOKEN
                )
                if success:
                    st.success(f"✅ Ticket submitted! [View on GitHub]({result})")
                else:
                    st.error(f"❌ Failed to submit ticket: {result}")
