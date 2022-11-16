import streamlit as st
import numpy as np
import pandas as pd
import os.path # used to know file extension
import IPython.display as ipd
import functions 

st.set_page_config(page_title= "Equalizer", layout="wide" ,page_icon=":musical_keyboard:")
st.markdown("<h1 style='text-align: center; color:darkcyan;'>Signal Equalizer</h1>", unsafe_allow_html=True)
with open("Equalizer.css")as source_des:
    st.markdown(f"<style>{source_des.read()} </style>", unsafe_allow_html=True)

if 'start' not in st.session_state:
    st.session_state['start']=0
if 'size1' not in st.session_state:
    st.session_state['size1']=0
if 'flag' not in st.session_state:
    st.session_state['flag'] = 0
#------------------------------------------------------------------Upload_file----------------------------------------------------------------------------------------------------------------------------------------------

option = st.sidebar.selectbox("Pick your sample!", options=[ "Uniform Range Mode", "Vowels Mode", "Musical Instruments Mode", "Biological Signal Abnormalities"])


uploaded_file = st.sidebar.file_uploader("uploader",key="uploaded_file",label_visibility="hidden")

data=[]
    
if uploaded_file is not None:
    file_name=uploaded_file.name
    ext = os.path.splitext(file_name)[1][1:]

#---------------------------------------------------------------------csv----------------------------------------------------------------------------------------------------------------------------------------------
    if ext=='csv':
        df = pd.read_csv(uploaded_file)
        list_of_columns=df.columns
        time = df[list_of_columns[0]].to_numpy()
        data = df[list_of_columns[1]].to_numpy()
        fmax=functions.getFMax(time,data)
        sample_frequency=2*fmax
        samplerate=1/sample_frequency
        duration = len(time)/samplerate 
        
#----------------------------------------------------------------------wav----------------------------------------------------------------------------------------------------------------------------------------------
    elif ext=='wav':
        data, sample_frequency  = functions.handle_uploaded_audio_file(uploaded_file)
        samplerate=1/sample_frequency
        fmax=sample_frequency/2
        duration = len(data)/samplerate 
        time = np.linspace(0,duration,len(data))
        st.sidebar.markdown('# Original Signal')
        st.sidebar.audio(file_name)
        
elif option == "Biological Signal Abnormalities":
    data,time,sample_frequency=functions.arrhythima()
    samplerate=1/sample_frequency
    duration = time
#----------------------------------------------------------------------Sliders-------------------------------------------------------------------------------------------------------------------------------------------------------

# else:
#     functions.generate_sliders(bin_max_frequency_value=10,slidersNum=10)
#----------------------------------------------------------------------Fourier-------------------------------------------------------------------------------------------------------------------------------------------------------
flag=True
if not data==[]:
    fft_sig, amplitude,phase,frequencies=functions.Fourier_transform(data,sample_frequency)
    st.write(frequencies,amplitude)
    st.write(sample_frequency,samplerate,fmax)

#----------------------------------------------------------------------Musical Instruments Mode-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       
    if  option=='Musical Instruments Mode' :
        slidersNum=3
        flag=False
        
#-----------------------------------------------------------------------Vowels---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif option=='Vowels Mode':
        slidersNum=10
        
#----------------------------------------------------------------------Biological Signal Abnormalities-------------------------------------------------------------------------------------------------------------------------------------------------------
    elif option=='Biological Signal Abnormalities':
        slidersNum=4
        
#-----------------------------------------------------------------------Uniform Range Mode-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif option=='Uniform Range Mode':
        slidersNum=10
        

#-------------------------------------------------------Bins_separation/generate sliders/signal-modification--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    points_per_freq=len(frequencies) /fmax
    freq_axis_list, amplitude_axis_list,bin_max_frequency_value=functions.bins_separation(frequencies, amplitude ,slidersNum)
    sliders_data= functions.generate_sliders(fmax,frequencies,points_per_freq,bin_max_frequency_value,slidersNum,flag)

    if  option=='Musical Instruments Mode' :
        mod_amplitude_axis_list,empty= functions.instruments(amplitude,frequencies,fmax,sliders_data)
    else:
        mod_amplitude_axis_list,empty= functions.signal_modification(sliders_data,amplitude,slidersNum,frequencies,fmax,bin_max_frequency_value,freq_axis_list)
#------------------------------------------------------------------------Static-plotting--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    functions.plot_signal(time,data,frequencies,amplitude) #time-domain representation, This shows us the loudness (amplitude) of sound wave changing with time.    

#------------------------------------------------------------------------Inverse-Fourier-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # inverse_button = st.sidebar.button("Display")
    # if inverse_button :
    phase=phase[:len(mod_amplitude_axis_list):1]
    ifft_file = functions.inverse_fourier(mod_amplitude_axis_list,phase)
    modified_time_axis = np.linspace(0, duration, len(mod_amplitude_axis_list))

    if option == 'Musical Instruments Mode' or 'Vowels Mode':
        st.sidebar.markdown('# Modified Signal')
        modified_audio = ipd.Audio(ifft_file, rate=sample_frequency/2)
        st.sidebar.write(modified_audio)


#---------------------------------------------------------------------------Spectrogram----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.markdown('## Spectrogram')
    spec1 = st.sidebar.checkbox("Show", key=2)
    if spec1:
        functions.plot_spectrogram(data,mod_amplitude_axis_list,sample_frequency)

#------------------------------------------------------------------------Dynamic-Plotting-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        start_btn  = st.button("start")
        pause_btn  = st.button("pause")
        resume_btn = st.button("resume")
        # if inverse_button :
        data = data[:len(ifft_file)]
        functions.plotShow(data,ifft_file, start_btn,pause_btn,resume_btn, sample_frequency)






















