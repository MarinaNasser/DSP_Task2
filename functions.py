import streamlit as st
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
# from scipy.fftpack import fft
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import  streamlit_vertical_slider  as svs
import librosa
import librosa.display
import itertools
from plotly.subplots import make_subplots
import plotly.graph_objects as go


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def Change_play_State():
    st.session_state['play_state']=  not st.session_state['play_state']

#--------------------------------------------------------------------------get Fmax----------------------------------------------------------------
def getFMax(xAxis,yAxis):
    amplitude = np.abs(sc.fft.rfft(yAxis))
    frequency = sc.fft.rfftfreq(len(xAxis), (xAxis[1]-xAxis[0]))
    indices = find_peaks(amplitude)
    if len(indices[0])>0 :
        max_freq=round(frequency[indices[0][-1]])
    else:
        max_freq=1   
    return max_freq

#-----------------------------------------------------------------------Read the Audiofile-----------------------------------------------------------------------------------------------------------------------------
 
def handle_uploaded_audio_file(uploaded_file):
    samples, sample_rate=librosa.load(uploaded_file, sr=None, mono=True, offset=0.0, duration=None)
    return samples, sample_rate
   
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_signal(time,data,freq,amp):

    SignalFigure, SignalAxis = plt.subplots(1, 2,figsize=(30, 10))
    SignalAxis[0].plot(time,data)
    SignalAxis[1].plot(freq,amp)
    SignalAxis[0].set_xlabel(xlabel='Time [sec]', size=25)
    SignalAxis[0].set_ylabel(ylabel='Amplitude', size=25)
    SignalAxis[0].set_title("Time representation", fontsize=30)
    
    SignalAxis[1].set_xlabel(xlabel='Frequency [sec]', size=25)
    SignalAxis[1].set_ylabel(ylabel='Amplitude [dB]', size=25)
    SignalAxis[1].set_title("Frequency representation", fontsize=30)
    
    st.pyplot(SignalFigure)

#  ----------------------------------------------------------------------------------------------------------------------------------------------
# get the fourier transform of the file
def Fourier_transform(data, samplerate):
    sampling_frequency=1/samplerate
    fft_sig = np.fft.fft(data)/len(data)  # Normalize data
    fft_sig = fft_sig[range(int(len(data)/2))] # Exclude sampling frequency
    amplitude= np.abs(fft_sig)
    phase =np.angle(fft_sig) # return the angle of the complex argument
    # sample_frequency =sc.fft.rfftfreq(len(data),d=time_step)  #return the discrete fourier transform sample frequencies
    length_of_data=len(data)
    values      = np.arange(int(length_of_data/2))
    timePeriod  = length_of_data/sampling_frequency
    frequencies = values/timePeriod

    return fft_sig, amplitude,phase,frequencies

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def bins_separation(frequency, amplitude, slidersNum):
    freq_axis_list = []
    amplitude_axis_list = []
    bin_max_frequency_value = int(len(frequency)/slidersNum) # 50 60 70 80 90 100 120  len()=20  int(20/10)=2  
    i = 0
    while(i < slidersNum):
        freq_axis_list.append(
            frequency[i*bin_max_frequency_value : (i+1)*bin_max_frequency_value])
        amplitude_axis_list.append(
            amplitude[i*bin_max_frequency_value : (i+1)*bin_max_frequency_value])
        i = i+1
    return freq_axis_list, amplitude_axis_list,bin_max_frequency_value
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# geberate sliders based on freq of uploaded file
def generate_sliders(bin_max_frequency_value , slidersNum):
        min_value=0
        max_value=0
        sliders_data = []
        boundary = int(50)
        columns = st.columns(slidersNum)
        for i in range(0, slidersNum):
            with columns[i]:
                min_value = 1- boundary
                max_value = 1 + boundary
                frequency_val = (i+1)*bin_max_frequency_value
                slider=svs.vertical_slider(key=i, default_value=1, step=1, min_value=min_value, max_value=max_value)
                st.write(f" { frequency_val } HZ")
                if slider == None:
                    slider = 1
                sliders_data.append(slider)
        return sliders_data


def signal_modification(sliders_data , List_amplitude_axis,slidersNum):
    empty = st.empty()
    empty.empty()
    modified_bins=[]
    for i in range(0,slidersNum):  
        modified_bins.append( 10**(sliders_data[i]/20) * List_amplitude_axis[i])
    
    mod_amplitude_axis_list=list(itertools.chain.from_iterable(modified_bins))
 
    # st.write(mod_amplitude_axis_list)
    
    return mod_amplitude_axis_list,empty


def inverse_fourier(mod_amplitude_axis_list,phase):
    modified_signal=np.multiply(mod_amplitude_axis_list,np.exp(1j*phase))
    # modified_signal=mod_amplitude_axis_list*np.cos(phase) +1j*mod_amplitude_axis_list*np.sin(phase) #list of complex no
    ifft_file=sc.ifft(modified_signal)
    # ifft_file=np.fft.ifft(modified_signal)
    return ifft_file


#-----------------------------------------------------------------spectrogram-----------------------------------------------------------------------------------------------------------------------------------
def plot_spectrogram(data,ifft_file,samplerate,mod_amplitude_axis_list):

    # yticks for spectrograms
    helper = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
    spec_yticks = [6.28 * i for i in helper]

    st.sidebar.markdown('## Spectrogram')
    spec1 = st.sidebar.checkbox("Show", key=2)
    
    fig2, ax = plt.subplots(1, 2, figsize=(30, 10))
   
    ax[0].specgram(data, Fs=samplerate)
    ax[0].set_xlabel(xlabel='Time [sec]', size=25)
    ax[0].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
    ax[0].set_yticks(helper)
    ax[0].set_yticklabels(spec_yticks)
    ax[0].set_title("Original signal", fontsize=30)
    ax[0].tick_params(axis='both', which='both', labelsize=18)

    ax[1].specgram(ifft_file, Fs=samplerate)
    ax[1].set_xlabel(xlabel='Time [sec]', size=25)
    ax[1].set_ylabel(ylabel='Frequency Amplitude [rad/s]', size=25)
    ax[1].set_yticks(helper)
    ax[1].set_yticklabels(spec_yticks)
    ax[1].set_title("Modified signal", fontsize=30)
    ax[1].tick_params(axis='both', which='both', labelsize=18)
    if spec1:
        st.pyplot(fig2)

   
