import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
from scipy.signal import find_peaks
import  streamlit_vertical_slider  as svs
import librosa
import librosa.display
import altair as alt
import time
from scipy.misc import electrocardiogram

#--------------------------------------------------------------------------Get Max Freq----------------------------------------------------------------
def getFMax(xAxis,yAxis):
    amplitude = np.abs(sc.fft.rfft(yAxis))
    frequency = sc.fft.rfftfreq(len(xAxis), (xAxis[1]-xAxis[0]))
    indices = find_peaks(amplitude)
    if len(indices[0])>0 :
        max_freq=round(frequency[indices[0][-1]])
    else:
        max_freq=1   
    return max_freq

#-----------------------------------------------------------------------Read the Audio File-----------------------------------------------------------------------------------------------------------------------------
 
def handle_uploaded_audio_file(uploaded_file):
    samples, sample_frequency=librosa.load(uploaded_file, sr=None, mono=True, offset=0.0, duration=None)
    return samples, sample_frequency
   
#----------------------------------------------------------------------Generate Sliders-------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_sliders(sliders_num,max_freq, mode):
        min_value=0
        max_value=0
        sliders_data = []
        Names=["Xylo", "Contrabass" , "Drums", "Flute", "Violin", "Trombone","Normal Sinus Rhythm","Abnormalities"," S"," Q"," M"]
        boundary = int(5)
        columns = st.columns(sliders_num)
        k=0
        for i in range(0, sliders_num):
            
            with columns[i]:
                min_value = - boundary
                max_value =  boundary
                frequency_val= int(max_freq/sliders_num)*(i+1)
                slider=svs.vertical_slider(key=i, default_value=1, step=1, min_value=min_value, max_value=max_value
                )
                if mode==4 :
                    st.write(f" { frequency_val } HZ")
                else:
                    if sliders_num == 6: k=k
                    elif sliders_num == 2: k=6
                    else : k=8
                    with columns[i]:
                        st.write(Names[k+i])
                        
                if slider == None:
                    slider = 1
                sliders_data.append(slider)
        return sliders_data
#----------------------------------------------------------------------Fourier Transformrier-------------------------------------------------------------------------------------------------------------------------------------------------------
def Fourier_transform(data, sample_frequency):
    fft_sig = np.fft.fft(data) # Normalize data
    fft_sig = fft_sig[range(int(len(data)/2))] # Exclude sampling frequency
    amplitude= np.abs(fft_sig)
    phase =np.angle(fft_sig) # return the angle of the complex argument
    length_of_data=len(data)
    values      = np.arange(int(length_of_data/2))
    timePeriod  = length_of_data/sample_frequency
    frequencies = values/timePeriod

    return fft_sig, amplitude,phase,frequencies

#----------------------------------------------------------------------Signal Modification-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       
def signal_modification(points_per_freq,max_freq,sliders_num,amplitude,sliders_data,mode):
    empty = st.empty()
    empty.empty()
    if mode==1:
        ranges = [[300,650,3500,6000],[700,3500],[0,700,600,700,6000,17000],[0,700],[700,2500],[2500,4000]]
        for instrumentIndex in range(len(ranges)):
            for index in range(0,len(ranges[instrumentIndex]),2):
                amplitude[int(ranges[instrumentIndex][index]*points_per_freq):int(ranges[instrumentIndex][index+1]*points_per_freq)]*=sliders_data[instrumentIndex]

       
    else:
        target_freq=max_freq/sliders_num
        for i in range(0,sliders_num):  
            amplitude[int(target_freq*(i)*points_per_freq) :int(target_freq*(i+1)*points_per_freq)]*=sliders_data[i]
        
   
    return amplitude,empty  
#-----------------------------------------------------------------Inverse Fourier-----------------------------------------------------------------------------------------------------------------------------------
def inverse_fourier(mod_amplitude_axis_list,phase):
    modified_signal=np.multiply(mod_amplitude_axis_list,np.exp(1j*phase))
    ifft_file=np.float64(np.fft.irfft(modified_signal))
    return ifft_file
#------------------------------------------------------------------------Static Plotting--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_signal(time,data,fft_time,ifft_file,frequencies,amplitude):
        # amp_dbs = 20 * np.log10(np.abs(amp))
        SignalFigure, SignalAxis = plt.subplots(1, 2,figsize=(40, 10))
        SignalAxis[0].plot(time,data)
        SignalAxis[1].plot(fft_time,ifft_file)
        SignalAxis[0].set_xlabel(xlabel='Time [sec]', size=25)
        SignalAxis[0].set_ylabel(ylabel='Amplitude', size=25)
        SignalAxis[0].set_title("Orignal Signal", fontsize=30)

        SignalAxis[1].set_xlabel(xlabel='Time [sec]', size=25)
        SignalAxis[1].set_ylabel(ylabel='Amplitude [dB]', size=25)
        SignalAxis[1].set_title("Modified Signal", fontsize=30)

        st.pyplot(SignalFigure)

#----------------------------------------------------------------------Dynamic Plotting-------------------------------------------------------------------------------------------------------------------------------------------------------

def plotShow(data, idata,resume_btn,sr):
    time1 = len(data)/(sr)
    if time1>1:
        time1 = int(time1)
    time1 = np.linspace(0,time1,len(data))   
    df = pd.DataFrame({'time': time1[::300], 
                        'amplitude': data[:: 300],
                        'amplitude after processing': idata[::300]}, columns=[
                        'time', 'amplitude','amplitude after processing'])
    N = df.shape[0]  # number of elements in the dataframe
    burst = 10      
    size = burst 
    step_df = df.iloc[:st.session_state.size1]
    if st.session_state.size1 ==0:
        step_df = df.iloc[0:N]
    lines = plot_animation(step_df)
    line_plot = st.altair_chart(lines)
    line_plot= line_plot.altair_chart(lines)
    
    if resume_btn: 
        st.session_state.flag = not(st.session_state.flag)
        if st.session_state.flag :
            for i in range( st.session_state.start,N):
                st.session_state.start =i 
                step_df = df.iloc[size:size+i]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                st.session_state.size1 = size
                size = i + burst
                time.sleep(.1)
    
    if st.session_state.flag :
        for i in range(st.session_state.start,N):
                st.session_state.start =i 
                step_df = df.iloc[size:size+i]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                st.session_state.size1 = size
                size = i + burst
                time.sleep(.1)

# Second Graph
def plot_animation(df):
    brush = alt.selection_interval()
    chart1 = alt.Chart(df).mark_line().encode(
            x=alt.X('time', axis=alt.Axis(title='Time')),
            # y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude')),
        ).properties(
            width=400,
            height=150
        ).add_selection(
            brush).interactive()
    figure = chart1.encode(
                y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude')))| chart1.encode(
                y=alt.Y('amplitude after processing',axis=alt.Axis(title='Amplitude after processing'))).add_selection(
            brush)
    return figure
#-----------------------------------------------------------------Spectrogram-----------------------------------------------------------------------------------------------------------------------------------
def plot_spectrogram(data,ifft_file,sample_frequency):

    fig2, ax = plt.subplots(1, 2, figsize=(30, 7))
   
    ax[0].specgram(data, Fs=sample_frequency)
    ax[0].set_xlabel(xlabel='Time [sec]', size=25)
    ax[0].set_ylabel(ylabel='Frequency [Hz]', size=25)
    ax[0].set_title("Original signal", fontsize=30)
    ax[0].tick_params(axis='both', which='both', labelsize=18)

    ax[1].specgram(ifft_file, Fs=sample_frequency)
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