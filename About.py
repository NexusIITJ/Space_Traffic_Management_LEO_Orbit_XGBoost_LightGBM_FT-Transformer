import streamlit as st

# --- Custom Styling ---
st.markdown("""
<style>
    .team-card {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30333d;
        text-align: center;
    }
    .project-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        color: #31333F;
        border-left: 5px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

st.header("About Our Mission")

# --- Project Description Section ---
st.subheader("🚀 Project Overview")
st.markdown("""
<div class="project-box">
    <h3>Space Traffic Management</h3>
    <h4>Ensuring Safe and Coordinated Operations in Orbit</h4>
    <p>
        As orbital congestion grows, the risk of collision cascades (Kessler Syndrome) increases. 
        Our project focuses on developing intelligent frameworks for real-time space situational awareness, 
        collision avoidance, and the sustainable management of orbital assets.
    </p>
</div>
""", unsafe_allow_html=True)

