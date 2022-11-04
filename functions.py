import streamlit as st
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.fftpack import fft
import  streamlit_vertical_slider  as svs



def Change_play_State():
    st.session_state['play_state']=  not st.session_state['play_state']


# def Audio_player(file):
#     st.audio(file, format="audio/wav", start_time=0)


# Read the Audiofile
def handle_uploaded_audio_file(uploaded_file):
    a = AudioSegment.from_wav(uploaded_file)
    samples = a.get_array_of_samples()
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples.typecode).max
    
    return fp_arr,  48000 #22050 

def show_signal(x_axis,y_axis):
    SignalFigure, SignalAxis = plt.subplots(1, 1)
    SignalAxis.plot(x_axis,y_axis)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    st.plotly_chart(SignalFigure,use_container_width=True)

# get the fourier transform of the file
def Fourier_transform(data, samplerate):
    fft_file = np.fft.fft(data)
    amplitude= np.abs(fft_file)
    phase =np.angle(fft_file)
    frequency =sc.fft.rfftfreq(len(data),1/samplerate)
    return amplitude,phase,frequency

def bins_separation(frequency, amplitude):
    freq_axis_list = []
    amplitude_axis_list = []
    bin_max_frequency_value = int(len(frequency)/10)
    i = 0
    while(i < 10):
        freq_axis_list.append(
            frequency[i*bin_max_frequency_value : (i+1)*bin_max_frequency_value])
        amplitude_axis_list.append(
            amplitude[i*bin_max_frequency_value:(i+1)*bin_max_frequency_value])
        i = i+1
    return freq_axis_list, amplitude_axis_list,bin_max_frequency_value

# geberate sliders based on freq of uploaded file
def generate_sliders(bin_max_frequency_value):
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
                var = (i+1)*bin_max_frequency_value
                slider1=svs.vertical_slider(key=key, default_value=1, step=1, min_value=min_value, max_value=max_value)
                st.write(f" { var } HZ")
                



