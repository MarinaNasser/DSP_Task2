import numpy as np
from scipy import fftpack
import streamlit as st
import  streamlit_vertical_slider  as svs
import pandas as pd
import functions as functions 
from scipy.io.wavfile import read
import os.path
import IPython.display as ipd
import scipy 

import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Equalizer")

#-----------------------------------------------------------------session state-------------------------------------------------------------------------------------------------------------------------------------
if 'play_state' not in st.session_state:
     st.session_state['play_state']= False


if 'uploaded' not in st.session_state:
     st.session_state['uploaded']= False

#------------------------------------------------------------------Upload_file----------------------------------------------------------------------------------------------------------------------------------------------
option = st.selectbox("Pick your sample!", options=["Take your pick", "Piano Music", "Biosignal", "Sine waves", "Vowels"])
if not option=="Take your pick":

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
            max_freq=functions.getFMax(time,data)
            samplerate=2*max_freq
        

    #------------------------------------------------------------------wav----------------------------------------------------------------------------------------------------------------------------------------------
        elif ext=='wav':
            # functions.Audio_player(uploaded_file)
            data, samplerate  = functions.handle_uploaded_audio_file(uploaded_file)
            duration = len(data)/samplerate
            time = np.arange(0,duration,1/samplerate)
            st.sidebar.markdown('# Original Signal')
            st.sidebar.audio(file_name)
            st.sidebar.markdown('# Modified Signal')
            
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        fft_sig, amplitude,phase,sample_frequency=functions.Fourier_transform(data,samplerate)
        freq_axis_list, amplitude_axis_list,bin_max_frequency_value=functions.bins_separation(sample_frequency, amplitude)
        # st.write(bin_max_frequency_value)
        sliders_data=functions.generate_sliders(bin_max_frequency_value)
        
        mod_amplitude_axis_list,empty= functions.sound_modification(sliders_data,amplitude_axis_list)
        # modified_time_axis=np.linspace(0, duration, len(mod_amplitude_axis_list))
        phase=phase[:len(mod_amplitude_axis_list):1]
        ifft_file=functions.inverse_fourier(mod_amplitude_axis_list,phase) 
        uploaded_file=ipd.Audio(ifft_file,rate=samplerate/2)
        empty.write(uploaded_file)
        frequency= sample_frequency[:len(mod_amplitude_axis_list):1]

        functions.show_signal(time,data) #plots wav file data in time domain
        functions.plot_spectrogram(data,ifft_file,samplerate,mod_amplitude_axis_list)