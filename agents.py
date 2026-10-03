def main() -> None:

    sidebar_manuals()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Troubleshooting Agent",
            "Training Agent",
            "Marine AI Command Center",
            "Learning",
        ],
    )

    try:

        if page == "Troubleshooting Agent":

            troubleshooting_page()

        elif page == "Training Agent":

            st.markdown(
                '<div class="main-title">'
                "Technical Training Agent"
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="main-subtitle">'
                "Professional technician learning materials powered by manual RAG + AI research"
                "</div>",
                unsafe_allow_html=True,
            )

            st.success(
                "MarineWise Training Agent is ready."
            )

            st.write(
                "The Training Agent can create professional "
                "training presentations, quizzes, assessment "
                "scores and remedial learning material."
            )

            tabs = st.tabs(
                [
                    "2A Professional Training",
                    "2B Quiz Generator",
                    "2C Score Assessment",
                ]
            )

            with tabs[0]:
                training_material_page()

            with tabs[1]:
                quiz_page()

            with tabs[2]:
                assessment_page()

        elif page == "Marine AI Command Center":

            marine_ai_command_center_page()

        else:

            learning_page()

    except Exception as exc:

        import traceback

        message = str(exc)

        st.error(
            f"MarineWise AI encountered an error: {exc}"
        )

        with st.expander(
            "Show technical error details"
        ):

            st.code(
                traceback.format_exc(),
                language="text",
            )

        if (
            "context_length_exceeded"
            in message
            or "reduce the length"
            in message.lower()
        ):

            st.error(
                f"The request was too large for the "
                f"current {selected_provider()} request limits."
            )

            st.info(
                "Troubleshooting uses a deliberately "
                "small context budget. Training uses a "
                "separate larger context budget."
            )

        else:

            st.caption(
                "Also check that the selected provider API key, "
                "TAVILY_API_KEY, and uploaded manual are configured "
                "correctly."
            )


if __name__ == "__main__":
    main()
