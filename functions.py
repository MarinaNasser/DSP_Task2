import streamlit as st
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.fftpack import fft
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
    a = AudioSegment.from_wav(uploaded_file)
    samples = a.get_array_of_samples()
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples.typecode).max
    
    return fp_arr,  48000 #22050 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def show_signal(x_axis,y_axis):
    SignalFigure, SignalAxis = plt.subplots(1, 1)
    SignalAxis.plot(x_axis,y_axis)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    st.plotly_chart(SignalFigure,use_container_width=True)
    #  ----------------------------------------------------------------------------------------------------------------------------------------------
# get the fourier transform of the file
def Fourier_transform(data, samplerate):
    time_step=1/samplerate
    fft_sig = np.fft.fft(data)
    amplitude= np.abs(fft_sig)
    phase =np.angle(fft_sig) #np.angle() return the angle of the complex argument
    sample_frequency =sc.fft.rfftfreq(len(data),d=time_step)  #return the discrete fourier transform sample frequencies
    return fft_sig, amplitude,phase,sample_frequency

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


def sound_modification(sliders_data , List_amplitude_axis):
    empty = st.empty()
    empty.empty()
    modified_bins=[]
    for i in range(0,4):
        modified_bins.append( 10**(sliders_data[i]/20) * List_amplitude_axis[i])
    
    mod_amplitude_axis_list=list(itertools.chain.from_iterable(modified_bins))
    return mod_amplitude_axis_list,empty


def inverse_fourier(mod_amplitude_axis_list,phase):
    modified_signal=np.multiply(mod_amplitude_axis_list,np.exp(1j*phase))
    ifft_file=sc.ifft(modified_signal)
    return ifft_file


#-----------------------------------------------------------------spectrogram-----------------------------------------------------------------------------------------------------------------------------------
def plot_spectrogram(data,ifft_file,samplerate,mod_amplitude_axis_list):

    # yticks for spectrograms
    helper = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
    spec_yticks = [6.28 * i for i in helper]

    st.sidebar.markdown('## Spectrogram')
    spec1 = st.sidebar.checkbox("Show", key=2)
    
    fig2, ax = plt.subplots(1, 2, figsize=(30, 15))
   
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

   
