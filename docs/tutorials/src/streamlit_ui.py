import json
import os

import requests
import streamlit as st

st.set_page_config(page_title="LLM Playground", layout="wide")

API_URL = os.getenv("API_URL")  # e.g. http://localhost:8000/v1/generate
assert API_URL, "API_URL not set"

# Options (from your Gradio code)
PROMPT_OPTIONS = ["finance_endpoint", "sport_endpoint"]
TONE_OPTIONS = ["formal", "casual", "optimistic", "neutral"]
DEPTH_LEVELS = ["basic overview", "detailed explanation", "expert-level analysis"]
USER_ID = 12345


def generate(model_name: str, tone: str, depth: str, prompt: str) -> str:
    """API call for model generation."""
    payload = {
        "model_name": model_name,
        "user_query": prompt,
        "response_detail_level": depth,
        "customer_id": USER_ID,
        "reply_style": tone,
    }
    resp = requests.post(API_URL, data=json.dumps(payload).encode("utf-8"))
    resp.raise_for_status()
    resp_json = resp.json()
    return (
        resp_json.get(model_name, {})
        .get("outputs", {})
        .get("answer", "No response available.")
    )


def ensure_state() -> None:
    """Initialize session state variables if not already set."""
    defaults = {
        "messages": [],
        "model_name": PROMPT_OPTIONS[0],
        "tone": TONE_OPTIONS[0],
        "depth": DEPTH_LEVELS[0],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar() -> None:
    """Sidebar controls for parameters and clearing chat."""
    with st.container(height=600):
        st.write("#### Parameters")
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=PROMPT_OPTIONS,
            index=PROMPT_OPTIONS.index(st.session_state.model_name),
        )
        st.session_state.tone = st.selectbox(
            "Select Tone",
            options=TONE_OPTIONS,
            index=TONE_OPTIONS.index(st.session_state.tone),
        )
        st.session_state.depth = st.selectbox(
            "Select Depth",
            options=DEPTH_LEVELS,
            index=DEPTH_LEVELS.index(st.session_state.depth),
        )

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat():
    """Render the chat interface and handle user input."""
    with st.container(height=600):
        messages = st.container(height=500)

        # Render prior chat
        for m in st.session_state.messages:
            role = m.get("role", "assistant")
            content = m.get("content", "")
            messages.chat_message(role).write(content)

        # Chat input
        user_prompt = st.chat_input("Type a question:")
        if user_prompt:
            # Show user message
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            messages.chat_message("user").write(user_prompt)

            # Call backend
            with messages:
                with st.spinner("Generating response..."):
                    bot_message = generate(
                        st.session_state.model_name,
                        st.session_state.tone,
                        st.session_state.depth,
                        user_prompt,
                    )

            # Show assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_message}
            )
            with messages.chat_message("assistant"):
                st.write(bot_message)

            st.rerun()


def main():
    ensure_state()
    st.write("# 🤖 LLM Playground with Model Selector")
    left, right = st.columns([3, 1], gap="small")
    with right:
        render_sidebar()
    with left:
        render_chat()


if __name__ == "__main__":
    main()
