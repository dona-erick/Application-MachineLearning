import streamlit as st

def profile():
    st.title("Mon Profil")
    st.write("Pour toute collaboration, veuillez me contacter :")

    # Style et liens de contact avec icônes
    st.markdown(
        """
        <style>
            .main {
                background: linear-gradient(to bottom right, #84fab0, #8fd3f4);
                color: #333333;
                padding: 1em;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                font-size: 1.1em;
            }
            .contact-links {
                margin-top: 1em;
            }
            .contact-links a {
                color: #333333;
                text-decoration: none;
                display: block;
                margin: 0.5em 0;
            }
            .contact-links a:hover {
                color: #0056b3;
            }
        </style>
        <div class="main">
            <div class="contact-links">
                <a href="https://github.com/dona-erick">📂 Github</a>
                <a href="https://www.linkedin.com/in/dona-erick">💼 LinkedIn</a>
                <a href="mailto:donaerickoulodji@gmail.com">📧 Email</a>
            </div>
        </div>
        <div style="text-align: center; padding: 1em 0; color: #666666; font-size: 0.9em;">
        © 2024 KOULODJI Dona Eric. Tous droits réservés.
        </div>
        """,
        unsafe_allow_html=True
    )