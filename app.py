import streamlit as st
import requests

st.title("Local AI Document Chat 🤖")

# User input
user_input = st.text_input("Ask a question about the document:")

if st.button("Send"):
    try:
        response = requests.get(
            "http://127.0.0.1:8000/chat",
            params={"prompt": user_input}
        )
        data = response.json()
        answer = data.get("response", "No answer returned.")

        # Show AI answer
        st.subheader("AI Answer:")
        st.write(answer)

        # Optional: Show context used (debug)
        # if "context" in data:
        #     st.subheader("Context Sent to AI (Debug):")
        #     st.text(data["context"])

    except Exception as e:
        st.write("Backend not running or error occurred:", e)