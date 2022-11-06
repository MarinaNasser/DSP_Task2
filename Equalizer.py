import numpy as np
from scipy import fftpack
import streamlit as st
import  streamlit_vertical_slider  as svs
import pandas as pd
import functions as functions 
from scipy.io.wavfile import read
import os.path
import IPython.display as ipd



st.set_page_config(layout="wide")


#-----------------------------------------------------------------session state-------------------------------------------------------------------------------------------------------------------------------------
if 'play_state' not in st.session_state:
     st.session_state['play_state']= False


if 'uploaded' not in st.session_state:
     st.session_state['uploaded']= False

#------------------------------------------------------------------Upload_file----------------------------------------------------------------------------------------------------------------------------------------------

uploaded_file = st.sidebar.file_uploader("uploader",key="uploaded_file",label_visibility="hidden")

if uploaded_file is not None:
    st.session_state['uploaded']= True
    file_name=uploaded_file.name
    ext = os.path.splitext(file_name)[1][1:]
    # st.write(ext)

    data=[]
#------------------------------------------------------------------csv----------------------------------------------------------------------------------------------------------------------------------------------
    if ext=='csv':
        df = pd.read_csv(uploaded_file)
        list_of_columns=df.columns
        time = df[list_of_columns[0]].to_numpy()
        data = df[list_of_columns[1]].to_numpy()
        functions.show_signal(x_axis,data)
        max_freq=functions.getFMax(x_axis,data)
        samplerate=2*max_freq
       

#------------------------------------------------------------------wav----------------------------------------------------------------------------------------------------------------------------------------------
    elif ext=='wav':
        # functions.Audio_player(uploaded_file)
        data, samplerate  = functions.handle_uploaded_audio_file(uploaded_file)
        duration = len(data)/samplerate
        time = np.arange(0,duration,1/samplerate)
        st.write("Original Sound")
        st.audio(file_name)
        
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    fft_sig, amplitude,phase,sample_frequency=functions.Fourier_transform(data,samplerate)
    List_freq_axis, List_amplitude_axis,bin_max_frequency_value=functions.bins_separation(sample_frequency, amplitude)
    st.write(bin_max_frequency_value)
    sliders_data=functions.generate_sliders(bin_max_frequency_value)
    functions.show_signal(time,data) #plots wav file data in time domain
    functions.plot_spectrogram(data,samplerate)
