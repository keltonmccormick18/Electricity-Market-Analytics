import streamlit as st

st.set_page_config(
    page_title="Energy Grid Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

pg = st.navigation(
    {
        "Overview": [
            st.Page("views/home.py", title="Home", icon="🏠", default=True),
        ],
        "Analytics": [
            st.Page("views/generation_mix.py",  title="Generation Mix",  icon="⚡"),
            st.Page("views/price_analytics.py", title="Price Analytics", icon="💰"),
            st.Page("views/sql_explorer.py",    title="SQL Explorer",    icon="🔍"),
        ],
        "Forecasting": [
            st.Page("views/forecast.py", title="Demand Forecast", icon="🔮"),
        ],
    }
)

pg.run()
