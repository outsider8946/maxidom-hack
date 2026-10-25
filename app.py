"""Точка входа Streamlit: навигация между страницами приложения."""

import streamlit as st

st.set_page_config(page_title="Maxidom", page_icon="🧱", layout="wide")

page = st.navigation(
    [
        st.Page("pages/01_RAG.py", title="Подбор и смета", icon="🧱"),
        st.Page("pages/02_Matching.py", title="Матчинг", icon="🏷️"),
    ]
)
page.run()
