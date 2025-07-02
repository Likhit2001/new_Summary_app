import streamlit as st
import requests

st.set_page_config(page_title="SumItUp", page_icon="📝", layout="centered")

st.title("📝 SumItUp: One-Click Text Summarizer")
st.markdown("""
Welcome to **SumItUp** — a fast and simple summarizer that turns your paragraphs into key insights.

Just paste your paragraph below and click **Summarize**!
""")

st.markdown("---")
st.markdown("🔗 [Project Repository](https://github.com/your-username/sumitup-streamlit)")


API_URL = "https://your-api-endpoint.com/summarize" 

paragraph = st.text_area("Enter your paragraph here:", height=250)

if st.button("Summarize"):
    if paragraph.strip():
        with st.spinner("Generating summary..."):
            try:
                response = requests.post(API_URL, json={"text": paragraph})
                response.raise_for_status()
                summary = response.json().get("summary", "")
                st.success("Summary:")
                st.write(summary)
            except Exception as e:
                st.error("Something went wrong. Please try again.")
    else:
        st.warning("Please enter some text to summarize.")
