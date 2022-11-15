import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.signal import find_peaks
import  streamlit_vertical_slider  as svs
import librosa
import librosa.display
import itertools
import altair as alt
import time
from scipy.misc import electrocardiogram

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
   
#------------------------------------------------------------------------Static-plotting--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_signal(time,data,freq,amp):

        SignalFigure, SignalAxis = plt.subplots(1, 2,figsize=(30, 10))
        SignalAxis[0].plot(time,data)
        SignalAxis[1].plot(freq,amp)
        SignalAxis[0].set_xlabel(xlabel='Time [sec]', size=25)
        SignalAxis[0].set_ylabel(ylabel='Amp litude', size=25)
        SignalAxis[0].set_title("Time representation", fontsize=30)

        SignalAxis[1].set_xlabel(xlabel='Frequency [Hz]', size=25)
        SignalAxis[1].set_ylabel(ylabel='Amplitude [dB]', size=25)
        SignalAxis[1].set_title("Frequency representation", fontsize=30)

        st.pyplot(SignalFigure)

#----------------------------------------------------------------------Fourier-------------------------------------------------------------------------------------------------------------------------------------------------------
def Fourier_transform(data, samplerate):

    fft_sig = np.fft.fft(data)/len(data)  # Normalize data
    fft_sig = fft_sig[range(int(len(data)/2))] # Exclude sampling frequency
    amplitude= np.abs(fft_sig)
    phase =np.angle(fft_sig) # return the angle of the complex argument
    # frequencies =sc.fft.rfftfreq(len(data),d=1/samplerate)  #return the discrete fourier transform sample frequencies
    length_of_data=len(data)
    values      = np.arange(int(length_of_data/2))
    timePeriod  = length_of_data/samplerate
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
#----------------------------------------------------------------------Generate Sliders-------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_sliders(bin_max_frequency_value , slidersNum,flag=True):
        min_value=0
        max_value=0
        sliders_data = []
        boundary = int(50)
        columns = st.columns(slidersNum)
        for i in range(0, slidersNum):
            with columns[i]:
                min_value = - boundary
                max_value =  boundary
                frequency_val = (i+1)*bin_max_frequency_value
                slider=svs.vertical_slider(key=i, default_value=1, step=1, min_value=min_value, max_value=max_value)
                if flag:
                    st.write(f" { frequency_val } HZ")
                else:
                   with columns[0]:
                    st.write("Xylo")
                   with columns[1]:
                    st.write("Contrabass")
                   with columns[2]:
                    st.write("Drums")
                 
                if slider == None:
                    slider = 1
                sliders_data.append(slider)
        return sliders_data

#----------------------------------------------------------------------Dynamic Plotting-------------------------------------------------------------------------------------------------------------------------------------------------------

def altair_plot(original_df,modified_df):
    lines = alt.Chart(original_df).mark_line().encode(x=alt.X('0:T', axis=alt.Axis(title='Time')),y=alt.Y('1:Q', axis=alt.Axis(title='Amplitude'))).properties( width=400,height=300)
    modified_lines=alt.Chart(modified_df).mark_line().encode(x=alt.X('0:T', axis=alt.Axis(title='Time')),y=alt.Y('1:Q', axis=alt.Axis(title='Amplitude'))).properties(width=400,height=300).interactive()
    return lines
def plot_animation(original_df):
    lines = alt.Chart(original_df).mark_line().encode(x=alt.X('time', axis=alt.Axis(title='Time')),y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude')),).properties(width=400,height=300).interactive()
    return lines

def dynamic_plot(line_plot,original_df,modified_df):
    N = original_df.shape[0]  # number of elements in the dataframe
    burst = 6       # number of elements (months) to add to the plot
    size = burst    # size of the current dataset
    for i in range(1, N):
                step_df = original_df.iloc[0:size]
                mod_step_df = modified_df.iloc[0:size]
                lines = plot_animation(step_df)
                mod_lines=plot_animation(mod_step_df)
                concat=alt.hconcat(lines,mod_lines)
                line_plot = line_plot.altair_chart(concat)
                size = i + burst
                if size >= N:
                    size = N - 1
                time.sleep(.00000000001)

#----------------------------------------------------------------------Signal Modification-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       
    
def signal_modification(sliders_data , List_amplitude_axis,slidersNum):
    empty = st.empty()
    empty.empty()
    modified_bins=[]
    for i in range(0,slidersNum):  
        # modified_bins.append( 10**(sliders_data[i]/20) * List_amplitude_axis[i])
        modified_bins.append( sliders_data[i] * List_amplitude_axis[i])

    
    mod_amplitude_axis_list=list(itertools.chain.from_iterable(modified_bins))
    
    return mod_amplitude_axis_list,empty

#----------------------------------------------------------------------Musical Instruments-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------           

def instruments( amplitude, frequencies, fmax,sliders_data):
    empty = st.empty()
    empty.empty()
    points_per_freq=len(frequencies) /fmax    
    max_freq_xylo_1=650
    min_freq_xylo_1=300
    max_freq_xylo_2=6000
    min_freq_xylo_2=3500
    amplitude[int(min_freq_xylo_1*points_per_freq):int(max_freq_xylo_1*points_per_freq)]*=sliders_data[0]
    amplitude[int(min_freq_xylo_2*points_per_freq):int(max_freq_xylo_2*points_per_freq)]*=sliders_data[0]
    max_freq_contrabass=3500
    min_freq_contrabass=700
    amplitude[int(min_freq_contrabass*points_per_freq):int(max_freq_contrabass*points_per_freq)]*=sliders_data[1]
    max_freq_drums_1=17000
    min_freq_drums_1=6000
    max_freq_drums_2=300
    min_freq_drums_2=10
    max_freq_drums_3=700
    min_freq_drums_3=600
    amplitude[int(min_freq_drums_1*points_per_freq):int(max_freq_drums_1*points_per_freq)]*=sliders_data[2]
    amplitude[int(min_freq_drums_2*points_per_freq):int(max_freq_drums_2*points_per_freq)]*=sliders_data[2]
    amplitude[int(min_freq_drums_3*points_per_freq):int(max_freq_drums_3*points_per_freq)]*=sliders_data[2]

    return amplitude,empty

def inverse_fourier(mod_amplitude_axis_list,phase):


    modified_signal=np.multiply(mod_amplitude_axis_list,np.exp(1j*phase))
    # modified_signal=mod_amplitude_axis_list*np.cos(phase) +1j*mod_amplitude_axis_list*np.sin(phase) #list of complex no
    ifft_file=np.fft.ifft(modified_signal)
    # ifft_file=np.fft.ifft(modified_signal)
    return ifft_file


#-----------------------------------------------------------------spectrogram-----------------------------------------------------------------------------------------------------------------------------------
def plot_spectrogram(data,ifft_file,samplerate,mod_amplitude_axis_list):

    # yticks for spectrograms
    helper = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
    spec_yticks = [6.28 * i for i in helper]
    
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
    
    st.pyplot(fig2)

   
def arrhythima():

    ecg = electrocardiogram()       # Calling the arrhythmia database of a woman
    fs = 360                        # determining f sample
    time = np.arange(ecg.size) / fs # detrmining tima axis

    return ecg,time,fs