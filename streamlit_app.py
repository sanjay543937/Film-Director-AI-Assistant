
import streamlit as st

from app.agents.graph import ask_film_assistant


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Film Director AI Assistant",
    page_icon="🎬",
    layout="wide"
)


# -----------------------------
# SESSION STATE
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    st.title("🎬 Film Director AI")
    st.write(
        "Your AI assistant for filmmaking, "
        "cinematography and creative production."
    )

    st.divider()

    st.subheader("Capabilities")
    st.write("📚 PDF Knowledge (RAG)")
    st.write("🌐 Live Web Search")
    st.write("🔀 Intelligent Routing")

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# MAIN HEADER
# -----------------------------

st.title("🎬 Film Director AI Assistant")

st.caption(
    "Ask questions about filmmaking, camera techniques, "
    "lighting, cinematography and production."
)

st.divider()


# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            if message.get("route"):
                st.caption(
                    f"Selected route: {message['route'].upper()}"
                )

            if message.get("sources"):
                with st.expander("📚 PDF Sources"):
                    for source in message["sources"]:
                        st.write(
                            f"**File:** {source.get('source', 'Unknown')}"
                        )
                        st.write(
                            f"**Page:** {source.get('page', 'Unknown')}"
                        )

            if message.get("web_results"):
                with st.expander("🌐 Web Search Results"):
                    for result in message["web_results"]:
                        st.markdown(
                            f"**{result.get('title', 'Untitled')}**"
                        )

                        if result.get("url"):
                            st.markdown(
                                f"[Open source]({result['url']})"
                            )

                        st.write(result.get("content", ""))


# -----------------------------
# CHAT INPUT
# -----------------------------

question = st.chat_input(
    "Ask your filmmaking question..."
)


# -----------------------------
# PROCESS USER QUESTION
# -----------------------------

if question:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Call LangGraph backend
    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing your question..."
        ):

            try:
                result = ask_film_assistant(question)

                answer = result.get(
                    "answer",
                    "No answer was generated."
                )

                route = result.get("route", "unknown")

                sources = result.get("sources", [])

                web_results = result.get(
                    "web_results",
                    []
                )

                st.markdown(answer)

                st.caption(
                    f"Selected route: {route.upper()}"
                )

                if sources:

                    with st.expander("📚 PDF Sources"):
                        for source in sources:
                            st.write(
                                f"**File:** "
                                f"{source.get('source', 'Unknown')}"
                            )

                            st.write(
                                f"**Page:** "
                                f"{source.get('page', 'Unknown')}"
                            )

                if web_results:

                    with st.expander("🌐 Web Search Results"):
                        for item in web_results:

                            st.markdown(
                                f"**{item.get('title', 'Untitled')}**"
                            )

                            if item.get("url"):
                                st.markdown(
                                    f"[Open source]({item['url']})"
                                )

                            st.write(
                                item.get("content", "")
                            )

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "route": route,
                    "sources": sources,
                    "web_results": web_results
                })

            except Exception as error:

                error_message = (
                    f"Something went wrong: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })