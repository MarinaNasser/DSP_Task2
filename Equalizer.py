import numpy as np
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
        
# ----------------------------------------------------------------------fourier-------------------------------------------------------------------------------------------------------------------------------------------------------
time_step=0.05
time_vec=np.arange(0,10,time_step) #return evenly spaced time vector between [0,10]
period=5 
sig=(np.sin(2*np.pi*time_vec/period))+ 0.25*np.random.randn(time_vec.size) #for every 5*5 points, it completes a 1/4 cycle or every 20*5 points it completes 1 cycle +noise
sig_fft=fftpack.fft(sig)
amplitude=np.abs(sig_fft)
power=amplitude**2
phase=np.angle(sig_fft) #np.angle() return the angle of the complex argument
sample_freq=fftpack.fftfreq(sig.size, d=time_step) #return the discrete fourier transform sample frequencies
amp_freq=np.array([amplitude,sample_freq])
amp_position= amp_freq[0,:].argmax()
peak_freq = amp_freq[1, amp_position]
high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq)> peak_freq]=0
filtered_Sig=fftpack.ifft(high_freq_fft) # return discrete inverse fourier transform of real or complex sequence
