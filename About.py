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

# --- Team Section ---
st.divider()
st.subheader("👥 The Team")

# Using columns to display the team neatly
t1, t2 = st.columns(2)

with t1:
    st.markdown("**Shiv Yadav**")
    st.caption("IIT Jodhpur")
    
    st.markdown("**Chittanoor Sri Raghava**")
    st.caption("IIT Jodhpur")

with t2:
    st.markdown("**Pratham Karmakar**")
    st.caption("IIT Jodhpur")
    
    st.markdown("**Princy**")
    st.caption("IIT Jodhpur")

# --- Nexus Club Section ---
st.divider()
st.subheader("🔭 About Nexus Club")

col_logo, col_text = st.columns([1, 3])

with col_logo:
    # Using a generic telescope/astronomy icon as a placeholder for the Nexus logo
    st.image("https://cdn-icons-png.flaticon.com/512/4230/4230759.png", width=100)

with col_text:
    st.markdown("**The Astronomy & Space Science Society of IIT Jodhpur**")
    st.write("""
    Nexus is the hub for space enthusiasts at IIT Jodhpur. We foster a culture of engineering innovation 
    and scientific inquiry, working on advanced projects ranging from CanSat stabilization systems to 
    automated telescope designs. We aim to bridge the gap between theoretical astrophysics and practical space technology.
    """)

# --- Contact Section ---
st.divider()
st.markdown("### 📬 Contact Us")
st.write("For inquiries regarding our project or club activities:")
st.markdown("📧 **Email:** [astronomy@iitj.ac.in](mailto:astronomy@iitj.ac.in)")