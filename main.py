import streamlit as st
import numpy as np
import pandas as pd

main_page = st.Page("FrontPg.py",title="main Page" )
About = st.Page("About.py",title="About" )
Future_Development = st.Page("FD.py",title="Incoming development" )
pg = st.navigation([main_page,About,Future_Development])
pg.run()
