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
def generate_sliders(sliders_num,max_freq,flag=True):
        min_value=0
        max_value=0
        sliders_data = []
        Names=["Xylo", "Contrabass" , "Drums", "Flute", "Violin", "Trombone","Bradycardia","Normal Sinus Rhythm", "Sinus Tachyardia","Atrial Tachycatdia","S","Q"]
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
                if flag:
                    st.write(f" { frequency_val } HZ")
                else:
                    if sliders_num == 6: k=k
                    elif sliders_num == 4: k=6
                    else : k=10
                    with columns[i]:
                        st.write(Names[k+i])
                        
                if slider == None:
                    slider = 1
                sliders_data.append(slider)
        return sliders_data
#----------------------------------------------------------------------FouFourier Transformrier-------------------------------------------------------------------------------------------------------------------------------------------------------
def Fourier_transform(data, sample_frequency):
    fft_sig = np.fft.fft(data)/len(data)  # Normalize data
    fft_sig = fft_sig[range(int(len(data)/2))] # Exclude sampling frequency
    amplitude= np.abs(fft_sig)
    phase =np.angle(fft_sig) # return the angle of the complex argument
    length_of_data=len(data)
    values      = np.arange(int(length_of_data/2))
    timePeriod  = length_of_data/sample_frequency
    frequencies = values/timePeriod

    return fft_sig, amplitude,phase,frequencies

#----------------------------------------------------------------------Signal Modification-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       
def signal_modification(points_per_freq,max_freq,sliders_num,amplitude,sliders_data):
    empty = st.empty()
    empty.empty()
    # points_per_freq=len(frequencies) /max_freq
    for i in range(0,sliders_num):  
        amplitude[int((max_freq/sliders_num)*(i)*points_per_freq) :int((max_freq/sliders_num)*(i+1)*points_per_freq)]*=sliders_data[i]
    return amplitude,empty
# ----------------------------------------------------------------------Musical Instruments Modification-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------           
def instruments_modification(points_per_freq, amplitude,sliders_data):
    empty = st.empty()
    empty.empty()
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
    Drums_range=[0,700,600,700,6000,17000]
    i=0
    while i<len(Drums_range):
        amplitude[int(Drums_range[i]*points_per_freq):int(Drums_range[i+1]*points_per_freq)]*=sliders_data[2]
        i+=2
    #------------------Flute---------------
    Flute_range=[0,700]
    f=0
    while f<len(Flute_range):
        amplitude[int(Flute_range[f]*points_per_freq):int(Flute_range[f+1]*points_per_freq)]*=sliders_data[3]
        f+=2
    #------------------Violin---------------
    Violin_range=[700,2500]
    v=0
    while v<len(Violin_range):
        amplitude[int(Violin_range[v]*points_per_freq):int(Violin_range[v+1]*points_per_freq)]*=sliders_data[4]
        v+=2
      #------------------Trombone---------------
    Trombone_range=[2500,4000]
    t=0
    while t<len(Trombone_range):
        amplitude[int(Trombone_range[t]*points_per_freq):int(Trombone_range[t+1]*points_per_freq)]*=sliders_data[5]
        t+=2
  
    return amplitude,empty

#-----------------------------------------------------------------Inverse Fourier-----------------------------------------------------------------------------------------------------------------------------------
def inverse_fourier(mod_amplitude_axis_list,phase):
    modified_signal=np.multiply(mod_amplitude_axis_list,np.exp(1j*phase))
    # ifft_file=sc.ifft(modified_signal)
    ifft_file=np.fft.irfft(modified_signal)
    return ifft_file
#------------------------------------------------------------------------Static Plotting--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_signal(time,data,fft_time,ifft_file,frequencies,amplitude):
        # amp_dbs = 20 * np.log10(np.abs(amp))
        SignalFigure, SignalAxis = plt.subplots(1, 2,figsize=(40, 10))
        SignalAxis[0].plot(time,data)
        SignalAxis[1].plot(fft_time,ifft_file)
        SignalAxis[0].set_xlabel(xlabel='Time [sec]', size=25)
        SignalAxis[0].set_ylabel(ylabel='Amplitude', size=25)
        SignalAxis[0].set_title("Time representation", fontsize=30)

        SignalAxis[1].set_xlabel(xlabel='Frequency [Hz]', size=25)
        SignalAxis[1].set_ylabel(ylabel='Amplitude [dB]', size=25)
        SignalAxis[1].set_title("Frequency representation", fontsize=30)

        st.pyplot(SignalFigure)

#----------------------------------------------------------------------Dynamic Plotting-------------------------------------------------------------------------------------------------------------------------------------------------------

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
    N = df.shape[0]  
    burst = 10      
    size = burst 
    
    step_df = df.iloc[0:st.session_state.size1]
    if st.session_state.size1 ==0:
        step_df = df.iloc[0:N]

    lines = plot_animation(step_df)
    line_plot = st.altair_chart(lines)
    line_plot= line_plot.altair_chart(lines)

    N = df.shape[0]  
    burst = 10      
    size = burst    
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

#-----------------------------------------------------------------Spectrogram-----------------------------------------------------------------------------------------------------------------------------------
def plot_spectrogram(data,ifft_file,sample_frequency):

    fig2, ax = plt.subplots(1, 2, figsize=(30, 10))
   
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