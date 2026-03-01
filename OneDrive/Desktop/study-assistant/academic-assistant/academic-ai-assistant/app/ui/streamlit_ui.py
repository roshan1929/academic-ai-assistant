
"""Streamlit UI placeholders.

This module imports Streamlit only inside the runtime function so the
package can be imported even when Streamlit is not installed.
"""

from typing import Optional, Tuple


def launch_ui() -> Tuple[Optional[str], Optional[str]]:
    """Launch the UI and return (subject, user_input).

    Attempts to use Streamlit if available; otherwise falls back to a
    simple CLI prompt so the project remains runnable without
    additional dependencies.
    """
    try:
        import streamlit as st
    except Exception:
        # CLI fallback when Streamlit isn't installed
        print("Streamlit not available; falling back to CLI input.")
        subjects = ["OS", "CN", "AI", "SPCC"]
        print("Available subjects:", ", ".join(subjects))
        while True:
            subject = input("Select subject (OS, CN, AI, SPCC): ").strip()
            if subject in subjects:
                break
            print("Invalid subject. Please choose one of:", ", ".join(subjects))
        user_input = input("Enter your query: ")
        print("Submit received (CLI)")
        return subject, user_input

    # Streamlit UI
    st.title("Academic Study Assistant")
    subject = st.selectbox("Subject", ["OS", "CN", "AI", "SPCC"])  # type: ignore
    user_input = st.text_input("Enter your query")  # type: ignore
    if st.button("Submit"):  # type: ignore
        st.success("System ready")  # show placeholder response
        st.write("Subject:", subject)
        st.write("Input:", user_input)
        return subject, user_input

    return None, None
