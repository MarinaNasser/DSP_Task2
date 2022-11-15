import streamlit as st
import numpy as np
import pandas as pd
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
    # st.write(len(frequency))
    # st.write(bin_max_frequency_value)
    i = 0
    while(i < slidersNum):
        freq_axis_list.append(
            frequency[i*bin_max_frequency_value : (i+1)*bin_max_frequency_value])
        amplitude_axis_list.append(
            amplitude[i*bin_max_frequency_value : (i+1)*bin_max_frequency_value])
        i = i+1
    return freq_axis_list, amplitude_axis_list,bin_max_frequency_value
#----------------------------------------------------------------------Generate Sliders-------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_sliders(bin_max_frequency_value,slidersNum,flag=True):
        min_value=0
        max_value=0
        sliders_data = []
        boundary = int(50)
        columns = st.columns(slidersNum)
        for i in range(0, slidersNum):
            with columns[i]:
                min_value = - boundary
                max_value =  boundary
                # frequency_val = frequency[int(points_per_freq*(i))]+1
                frequency_val = (i+1)*bin_max_frequency_value
                slider=svs.vertical_slider(key=i, default_value=1, step=1, min_value=min_value, max_value=max_value)
                if flag:
                    st.write(f" { frequency_val } HZ")
                else:
                    if i==0:
                        with columns[0]:
                            st.write("Xylo")
                    elif i==1:
                        with columns[1]:
                            st.write("Contrabass")
                    elif i==2:
                        with columns[2]:
                            st.write("Drums")
                 
                if slider == None:
                    slider = 1
                sliders_data.append(slider)
        return sliders_data

#----------------------------------------------------------------------Dynamic Plotting-------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_animation(df):
    brush = alt.selection_interval()
    chart1 = alt.Chart(df).mark_line().encode(
            x=alt.X('time', axis=alt.Axis(title='Time')),
            # y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude')),
        ).properties(
            width=500,
            height=300
        ).add_selection(
            brush).interactive()
    
    figure = chart1.encode(
                  y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude')))| chart1.encode(
                  y=alt.Y('amplitude after processing',axis=alt.Axis(title='Amplitude after'))).add_selection(
            brush)

    return figure




def plotShow(data, idata,start_btn,pause_btn,resume_btn,sr):

    time1 = len(data)/(sr)
    if time1>1:
        time1 = int(time1)
    time1 = np.linspace(0,time1,len(data))   
    df = pd.DataFrame({'time': time1[::300], 
                        'amplitude': data[:: 300],
                        'amplitude after processing': idata[::300]}, columns=[
                        'time', 'amplitude','amplitude after processing'])
    N = df.shape[0]  # number of elements in the dataframe
    burst = 10      # number of elements (months) to add to the plot
    size = burst 
    
    step_df = df.iloc[0:st.session_state.size1]
    if st.session_state.size1 ==0:
        step_df = df.iloc[0:N]

    lines = plot_animation(step_df)
    line_plot = st.altair_chart(lines)
    line_plot= line_plot.altair_chart(lines)

    # lines = plot_animation(df)
    # line_plot = st.altair_chart(lines)
    N = df.shape[0]  # number of elements in the dataframe
    burst = 10      # number of elements (months) to add to the plot
    size = burst    #   size of the current dataset
    if start_btn:
        st.session_state.flag = 1
        for i in range(1, N):
            st.session_state.start=i
            step_df = df.iloc[0:size]
            lines = plot_animation(step_df)
            line_plot = line_plot.altair_chart(lines)
            size = i + burst 
            st.session_state.size1 = size
            time.sleep(.1)

    elif resume_btn: 
            st.session_state.flag = 1
            for i in range( st.session_state.start,N):
                st.session_state.start =i 
                step_df = df.iloc[0:size]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                st.session_state.size1 = size
                size = i + burst
                time.sleep(.1)

    elif pause_btn:
            st.session_state.flag =0
            step_df = df.iloc[0:st.session_state.size1]
            lines = plot_animation(step_df)
            line_plot= line_plot.altair_chart(lines)



    if st.session_state.flag == 1:
        for i in range( st.session_state.start,N):
                st.session_state.start =i 
                step_df = df.iloc[0:size]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                st.session_state.size1 = size
                size = i + burst
                time.sleep(.1)



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


# def signal_modification(sliders_data,amplitude,slidersNum,frequencies,fmax,bin_max_frequency_value,freq_axis_list):
#     empty = st.empty()
#     empty.empty()
#     points_per_freq=len(frequencies) /fmax
#     for i in range(0,slidersNum):  
#         amplitude[(freq_axis_list[i*bin_max_frequency_value]*points_per_freq) : (freq_axis_list[(i+1)*bin_max_frequency_value]*points_per_freq)]*=sliders_data[0]
#     return amplitude,empty
#----------------------------------------------------------------------Musical Instruments-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------           

def instruments( amplitude, frequencies, fmax,sliders_data):
    empty = st.empty()
    empty.empty()
    points_per_freq=len(frequencies) /fmax

    #------------------Xylo---------------
    Xylo_range=[300,650,3500,6000]
    k=0
    while k<len(Xylo_range):
        amplitude[int(Xylo_range[k]*points_per_freq):int(Xylo_range[k+1]*points_per_freq)]*=sliders_data[0]
        k+=2

    #------------------contrabass---------------
    contrabass_range=[700,3500]
    j=0
    while j<len(contrabass_range):
        amplitude[int(contrabass_range[j]*points_per_freq):int(contrabass_range[j+1]*points_per_freq)]*=sliders_data[1]
        j+=2
    
    #------------------Drums---------------
    Drums_range=[10,300,600,700,6000,17000]
    i=0
    while i<len(Drums_range):
        amplitude[int(Drums_range[i]*points_per_freq):int(Drums_range[i+1]*points_per_freq)]*=sliders_data[2]
        i+=2
  
    return amplitude,empty



#-----------------------------------------------------------------Inverse Fourier-----------------------------------------------------------------------------------------------------------------------------------

def inverse_fourier(mod_amplitude_axis_list,phase):


    modified_signal=np.multiply(mod_amplitude_axis_list,np.exp(1j*phase))
    # modified_signal=mod_amplitude_axis_list*np.cos(phase) +1j*mod_amplitude_axis_list*np.sin(phase) #list of complex no
    ifft_file=np.fft.irfft(modified_signal)
    # ifft_file=np.fft.ifft(modified_signal)
    return ifft_file


#-----------------------------------------------------------------Spectrogram-----------------------------------------------------------------------------------------------------------------------------------
def plot_spectrogram(data,mod_amplitude_axis_list,sample_frequency):

    fig2, ax = plt.subplots(1, 2, figsize=(30, 10))
   
    ax[0].specgram(data, Fs=sample_frequency)
    ax[0].set_xlabel(xlabel='Time [sec]', size=25)
    ax[0].set_ylabel(ylabel='Frequency [Hz]', size=25)
    ax[0].set_title("Original signal", fontsize=30)
    ax[0].tick_params(axis='both', which='both', labelsize=18)

    ax[1].specgram(mod_amplitude_axis_list, Fs=sample_frequency)
    ax[1].set_xlabel(xlabel='Time [sec]', size=25)
    ax[1].set_ylabel(ylabel='Frequency [Hz]', size=25)
    ax[1].set_title("Modified signal", fontsize=30)
    ax[1].tick_params(axis='both', which='both', labelsize=18)
    
    st.pyplot(fig2)


#-----------------------------------------------------------------arrhythima-----------------------------------------------------------------------------------------------------------------------------------
  
def arrhythima():

    ecg = electrocardiogram()       # Calling the arrhythmia database of a woman
    fs = 360                        # determining f sample
    time = np.arange(ecg.size) / fs # detrmining tima axis

    return ecg,time,fs