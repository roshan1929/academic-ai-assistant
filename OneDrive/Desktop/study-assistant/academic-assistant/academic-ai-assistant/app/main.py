"""Entrypoint for the academic assistant (placeholder).

This module orchestrates the UI and the offline RAG pipeline. Heavy
dependencies are imported at runtime so the module can be imported and
inspected without requiring those packages to be installed.
"""

from app import config
from app.ui import streamlit_ui


def run_app() -> None:
    """Run the application: get UI input, call RAG, display response.

    Keeps UI and pipeline responsibilities separate: the UI collects
    subject and question, this function loads the RAG pipeline at
    runtime and displays results. Uses a Streamlit spinner when
    available and falls back to CLI output.
    """
    print(f"Starting {config.APP_NAME} (placeholder)")

    subject, user_input = streamlit_ui.launch_ui()

    # If no submission, nothing to do
    if not subject or not user_input:
        return

    # Import the RAG pipeline at runtime
    try:
        from app.offline.pipeline import rag_chain
    except Exception as exc:
        print("Failed to import RAG pipeline:", exc)
        return

    # Try to use Streamlit spinner/display if available
    try:
        import streamlit as st
        use_streamlit = True
    except Exception:
        st = None
        use_streamlit = False

    if use_streamlit:
        with st.spinner("Processing question with RAG..."):
            response = rag_chain.answer_question(subject, user_input)
        if response:
            st.subheader("Response")
            st.write(response)
        else:
            st.warning("No response generated.")
    else:
        print("Processing question with RAG...")
        response = rag_chain.answer_question(subject, user_input)
        print("System ready")
        if response:
            print("Response:\n", response)
        else:
            print("No response generated.")


if __name__ == "__main__":
    run_app()
