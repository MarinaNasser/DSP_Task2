import numpy
from scipy import fftpack
import streamlit as st
import  streamlit_vertical_slider  as svs
import pandas as pd
st.set_page_config(layout="wide")

# ------------------------------------------------------------------Upload_file----------------------------------------------------------------------------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV",type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    list_of_columns=df.columns
    x_axis = df[list_of_columns[0]].to_numpy()
    y_axis = df[list_of_columns[1]].to_numpy()

# -------------------------------------------------------------------sliders---------------------------------------------------------------------------------------------------------------------------------------------
	
min_value=0
max_value=0
boundary = int(50)
sliders = {}
adjusted_data = []
columns = st.columns(10)
for i in range(10):
    key=i
    min_value = 1- boundary
    max_value = 1 + boundary
    with columns[i]:
        slider1=svs.vertical_slider(key=key, default_value=1, step=1, min_value=min_value, max_value=max_value)
        