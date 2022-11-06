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

uploaded_file = st.file_uploader("uploader",key="uploaded_file",label_visibility="hidden")

if uploaded_file is not None:
    st.session_state['uploaded']= True
    file_name=uploaded_file.name
    ext = os.path.splitext(file_name)[1][1:]
    # st.write(ext)


    if ext=='csv':
        df = pd.read_csv(uploaded_file)
        list_of_columns=df.columns
        x_axis = df[list_of_columns[0]].to_numpy()
        y_axis = df[list_of_columns[1]].to_numpy()
        functions.show_signal(x_axis,y_axis)


    elif ext=='wav':
        # functions.Audio_player(uploaded_file)
        data, samplerate  = functions.handle_uploaded_audio_file(uploaded_file)
        duration = len(data)/samplerate
        time = np.arange(0,duration,1/samplerate)
        amplitude,phase,frequency=functions.Fourier_transform(data,samplerate)
        freq_axis_list, amplitude_axis_list,bin_max_frequency_value=functions.bins_separation(frequency, amplitude)
        #sliders_date=functions.generate_sliders(bin_max_frequency_value)
        # st.write(bin_max_frequency_value)
        sliders_data=functions.generate_sliders(bin_max_frequency_value)
        # mod_amplitude_axis_list,empty=functions.sound_modification(sliders_data,amplitude_axis_list)
        # phase=phase[:len(mod_amplitude_axis_list):1]
        # ifft_file=functions.inverse_fourier(mod_amplitude_axis_list,phase)    # generate = st.button('Generate')
        # uploaded_file=ipd.Audio(ifft_file,rate=samplerate/2)
        # empty.write(uploaded_file)
        # frequency=frequency[:len(mod_amplitude_axis_list):1]




        # st.write("Duration of Audio in Seconds", duration)
        # st.write("Duration of Audio in Minutes", duration/60)
        functions.show_signal(time,data)

#-----------------------------------------------------Play Button-------------------------------------------------------------------------------------------------------------------------------------------------

    st.button(label= "DISPLAY" ,#if not st.session_state["play_state"] else "PAUSE", 
    disabled= not st.session_state['uploaded'], on_click= functions.Change_play_State())

    if not st.session_state["play_state"]:
        st.audio(file_name)
 
#-------------------------------------------------------sliders---------------------------------------------------------------------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # Filter the beat at 3kHz
# filtered = uploaded_file.low_pass_filter(3000)
# # Mix our filtered beat with the new loop at -3dB
# final = filtered.overlay(loop2 - 3, loop=True)


 
#-------------------------------------------------------save--------------------------------------------------------------------------------------------------------