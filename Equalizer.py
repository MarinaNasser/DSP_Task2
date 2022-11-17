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
flag=True
    
if uploaded_file is not None:
    file_name=uploaded_file.name
    ext = os.path.splitext(file_name)[1][1:]
#---------------------------------------------------------------------csv----------------------------------------------------------------------------------------------------------------------------------------------
    if ext=='csv':
        df = pd.read_csv(uploaded_file)
        list_of_columns=df.columns
        time = df[list_of_columns[0]].to_numpy()
        data = df[list_of_columns[1]].to_numpy()
        max_freq=functions.getFMax(time,data)
        sample_frequency=2*max_freq
        sample_rate=1/sample_frequency
        
#------------------------------------------ ----------------------------wav----------------------------------------------------------------------------------------------------------------------------------------------
    elif ext=='wav':
        data, sample_frequency  = functions.handle_uploaded_audio_file(uploaded_file)
        sample_rate=1/sample_frequency
        max_freq=sample_frequency/2
        duration = len(data)/sample_frequency 
        time = np.linspace(0,duration,len(data))
        st.sidebar.markdown('# Original Signal')
        st.sidebar.audio(file_name)
#----------------------------------------------------------------------Sliders-------------------------------------------------------------------------------------------------------------------------------------------------------
elif option== 'Musical Instruments Mode' :
    functions.generate_sliders(6,10,1)
elif option== 'Biological Signal Abnormalities' :
    functions.generate_sliders(4,10,3)
elif option== 'Vowels Mode' :
    functions.generate_sliders(3,10,2)
else:
    functions.generate_sliders(10,10,4)
#----------------------------------------------------------------------Fourier-------------------------------------------------------------------------------------------------------------------------------------------------------
if not data==[]:
    fft_sig, amplitude,phase,frequencies=functions.Fourier_transform(data,sample_frequency)

#----------------------------------------------------------------------Musical Instruments Mode-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       
    if  option=='Musical Instruments Mode' :
        sliders_num=6
        mode=1
        
#-----------------------------------------------------------------------Vowels---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif option=='Vowels Mode':
        sliders_num=3
        mode=2
        
#----------------------------------------------------------------------Biological Signal Abnormalities-------------------------------------------------------------------------------------------------------------------------------------------------------
    elif option=='Biological Signal Abnormalities':
        sliders_num=4
        mode=3
        
#-----------------------------------------------------------------------Uniform Range Mode-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif option=='Uniform Range Mode':
        sliders_num=10
        mode=4
    
#-------------------------------------------------------generate sliders/signal-modification--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    points_per_freq=len(frequencies) /max_freq
    sliders_data= functions.generate_sliders(sliders_num,max_freq,mode)
    if ext=='wav':
        st.markdown('## Modified Signal')

    mod_amplitude_axis_list,empty= functions.signal_modification(points_per_freq,max_freq,sliders_num,amplitude,sliders_data,mode)
#------------------------------------------------------------------------Static-plotting--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # functions.plot_signal(time,data,frequencies,amplitude) #time-domain representation, This shows us the loudness (amplitude) of sound wave changing with time.    

#------------------------------------------------------------------------Inverse-Fourier-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    phase=phase[:len(mod_amplitude_axis_list):1]
    ifft_file = functions.inverse_fourier(mod_amplitude_axis_list,phase)
    duration_fft= len(ifft_file)/sample_frequency
    fft_time = np.linspace(0,duration_fft,len(ifft_file))
  
    if ext=='wav':
        modified_audio = ipd.Audio(ifft_file, rate=sample_frequency)
        empty.write(modified_audio)

    functions.plot_signal(time,data,fft_time,ifft_file,frequencies,amplitude)     
#---------------------------------------------------------------------------Spectrogram----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.markdown('## Spectrogram')
    spec1 = st.sidebar.checkbox("Show", key=2)
    if spec1:
        functions.plot_spectrogram(data,ifft_file,sample_frequency)

#------------------------------------------------------------------------Dynamic-Plotting-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        column1, column2, column3, column4, column5, column6, column7, column8, column9, column10, column11, column12 = st.columns([1,1,1,1,1,1,1,1,1,1,1,1])
        with column1:
            start_btn  = st.button("▶️")
        with column2:
            pause_btn  = st.button("⏸")
        with column3:
            resume_btn = st.button("⏯")
        data = data[:len(ifft_file)]
        functions.plotShow(data,ifft_file, start_btn,pause_btn,resume_btn, sample_frequency)






















