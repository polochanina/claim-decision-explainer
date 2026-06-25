import traceback

import streamlit as st

from streamlit_app.api_client import ExplainClient
from streamlit_app.config import LANGUAGE_KEYS, LANGUAGE_LABELS, PERSONA_KEYS, PERSONA_LABELS, UI_TEXT
from streamlit_app.dataset import ClaimSampler


class AppUI:
    def __init__(self, session: dict, sampler: ClaimSampler, client: ExplainClient):
        self.session = session
        self._sampler = sampler
        self._client = client

    def run(self) -> None:
        try:
            self.display_header()
            self.display_sidebar()
            self.display_main()
        except Exception as e:
            self.show_error(e)

    def display_header(self) -> None:
        st.title(UI_TEXT["title"])
        st.caption(UI_TEXT["caption"])

    def display_sidebar(self) -> None:
        with st.sidebar:
            st.header(UI_TEXT["sidebar_header"])

            source_labels = {"real": UI_TEXT["source_real"], "generated": UI_TEXT["source_generated"]}
            selected_source = st.selectbox(
                UI_TEXT["source_label"],
                list(source_labels.keys()),
                format_func=lambda key: source_labels[key],
            )
            if selected_source != self.session.get("source"):
                self.session["source"] = selected_source
                self.session["claim"] = None
                self.session["result"] = None

            country_options = [UI_TEXT["all_countries"]] + self._sampler.countries(self.session["source"])
            self.session["country"] = st.selectbox(UI_TEXT["country_label"], country_options)

            self.session["persona"] = st.selectbox(
                UI_TEXT["persona_label"],
                PERSONA_KEYS,
                format_func=lambda key: PERSONA_LABELS.get(key, key),
            )

            self.session["language"] = st.selectbox(
                UI_TEXT["language_label"],
                LANGUAGE_KEYS,
                format_func=lambda key: LANGUAGE_LABELS.get(key, key),
            )

            if st.button(UI_TEXT["sample_button"]):
                self._handle_sample()

    def display_main(self) -> None:
        claim = self.session.get("claim")
        if not claim:
            st.info(UI_TEXT["no_claim_warning"])
            return

        st.subheader(UI_TEXT["claim_header"])
        st.json(claim)

        if st.button(UI_TEXT["explain_button"]):
            self._handle_explain(claim)

        self._display_result()

    def _handle_sample(self) -> None:
        country = self.session["country"]
        country = None if country == UI_TEXT["all_countries"] else country
        claim = self._sampler.sample(self.session["source"], country)
        self.session["claim"] = claim or None
        self.session["result"] = None
        if not claim:
            st.warning(UI_TEXT["no_claims_for_country_warning"])

    def _handle_explain(self, claim: dict) -> None:
        request = {**claim, "language": self.session["language"]}
        with st.spinner(UI_TEXT["spinner_explaining"]):
            self.session["result"] = self._client.explain(request)

    def _display_result(self) -> None:
        result = self.session.get("result")
        if not result:
            return

        decision_label = (
            UI_TEXT["decision_approved"] if result["decision"] == "approved" else UI_TEXT["decision_declined"]
        )
        col1, col2 = st.columns(2)
        col1.metric(UI_TEXT["decision_label"], decision_label)
        col2.metric(UI_TEXT["probability_label"], f"{result['approval_probability']:.1%}")

        st.subheader(UI_TEXT["factors_header"])
        st.dataframe(result["contributing_factors"], use_container_width=True)

        st.subheader(UI_TEXT["explanation_header"])
        persona = self.session["persona"]
        explanation = next(
            (item["explanation"] for item in result["explanations"] if item["persona"] == persona), None
        )
        if explanation is None:
            st.warning(UI_TEXT["no_explanation_for_persona_warning"])
        else:
            st.markdown(explanation)

    def show_error(self, e: Exception) -> None:
        st.error(UI_TEXT["error_title"])
        with st.expander("Details"):
            st.code(traceback.format_exc())
