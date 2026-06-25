import streamlit as st

from streamlit_app.api_client import ExplainClient
from streamlit_app.config import API_BASE_URL, APP_ICON, APP_NAME, DATASET_SOURCES
from streamlit_app.dataset import ClaimSampler
from streamlit_app.ui import AppUI

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide")

if "initialised" not in st.session_state:
    st.session_state["ui"] = AppUI(
        session=st.session_state,
        sampler=ClaimSampler(DATASET_SOURCES),
        client=ExplainClient(API_BASE_URL),
    )
    st.session_state["initialised"] = True

st.session_state["ui"].run()
