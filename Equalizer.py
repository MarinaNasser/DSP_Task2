import numpy as np
from scipy import fftpack
import streamlit as st
import  streamlit_vertical_slider  as svs
import pandas as pd
import functions 
from pydub import AudioSegment
from pydub.playback import play
from scipy.io.wavfile import read
import os.path
import matplotlib as plt
from scipy.fft import irfft
from scipy.io.wavfile import read,write

st.set_page_config(layout="wide")


#-----------------------------------------------------------------session state-------------------------------------------------------------------------------------------------------------------------------------
if 'play_state' not in st.session_state:
     st.session_state['play_state']= False


if 'uploaded' not in st.session_state:
     st.session_state['uploaded']= False
#------------------------------------------------------------------Upload_file----------------------------------------------------------------------------------------------------------------------------------------------

uploaded_file = st.file_uploader("uploader",key="uploaded_file",label_visibility="hidden")
functions.generate_sliders(100)

if uploaded_file is not None:
    st.session_state['uploaded']= True
    file_name=uploaded_file.name
    ext = os.path.splitext(file_name)[1][1:]
    # st.write(ext)

#------------------------------------------------------------------CSV----------------------------------------------------------------------------------------------------------------------------------------------

    if ext=='csv':
        df = pd.read_csv(uploaded_file)
        list_of_columns=df.columns
        x_axis = df[list_of_columns[0]].to_numpy()
        y_axis = df[list_of_columns[1]].to_numpy()
        functions.show_signal(x_axis,y_axis)

#------------------------------------------------------------------wav----------------------------------------------------------------------------------------------------------------------------------------------
    elif ext=='wav':
        # functions.Audio_player(uploaded_file)
        data, samplerate  = functions.handle_uploaded_audio_file(uploaded_file)
        duration = len(data)/samplerate
        time = np.arange(0,duration,1/samplerate)
        fft_sig, amplitude,phase,sample_frequency=functions.Fourier_transform(data,samplerate)
        List_freq_axis, List_amplitude_axis,bin_max_frequency_value=functions.bins_separation(sample_frequency, amplitude)
        #sliders_date=functions.generate_sliders(bin_max_frequency_value)
        st.write(bin_max_frequency_value)
        sliders_data=functions.generate_sliders(bin_max_frequency_value)
        st.write("Original Sound")
        st.audio(file_name)
        functions.show_signal(time,data) #plots wav file data in time domain
        