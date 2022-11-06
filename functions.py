import streamlit as st
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.fftpack import fft
import  streamlit_vertical_slider  as svs
import itertools



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
        sliders_data = []
        boundary = int(50)
        sliders = {}
        adjusted_data = []
        columns = st.columns(10)
        for i in range(0, 10):
            with columns[i]:
                min_value = 1- boundary
                max_value = 1 + boundary
                var = (i+1)*bin_max_frequency_value
                slider1=svs.vertical_slider(key=i, default_value=1, step=1, min_value=min_value, max_value=max_value)
                st.write(f" { var } HZ")
                if var == None:
                    var = 1
                sliders_data.append(slider1)
        return sliders_data

# def sound_modification(sliders_data,amplitude_axis_list):
#     empty = st.empty()
#     empty.empty()
#     modified_bins=[]
#     for i in range(0,10):
#         modified_bins.append( 10**(sliders_data[i]/20) * amplitude_axis_list[i])
#     st.write(modified_bins)
#     st.write(sliders_data)
#     mod_amplitude_axis_list=list(itertools.chain.from_iterable(modified_bins))
#     return mod_amplitude_axis_list,empty

# def inverse_fourier(mod_amplitude_axis_list,phase):
#     amplitude_modified=np.multiply(mod_amplitude_axis_list,np.exp(1j*phase))
#     ifft_file=sc.ifft(amplitude_modified)
#     return ifft_file