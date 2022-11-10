import numpy as np
from scipy import fftpack
import streamlit as st
import  streamlit_vertical_slider  as svs
import pandas as pd
import functions as functions 
from scipy.io.wavfile import read
import os.path
import IPython.display as ipd
from scipy.io import wavfile
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt

st.set_page_config(page_title= "Equalizer", layout="wide" ,page_icon=":musical_keyboard:")

with open("Equalizer.css")as source_des:
    st.markdown(f"<style>{source_des.read()} </style>", unsafe_allow_html=True)

#------------------------------------------------------------------Upload_file----------------------------------------------------------------------------------------------------------------------------------------------

option = st.sidebar.selectbox("Pick your sample!", options=[ "Music", "Biosignal", "Sine waves", "Vowels"])

if not option=="Take your pick":

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
            max_freq=functions.getFMax(time,data)
            samplerate=2*max_freq
            duration = len(time)/samplerate 
        
#----------------------------------------------------------------------wav----------------------------------------------------------------------------------------------------------------------------------------------
        elif ext=='wav':
            # functions.Audio_player(uploaded_file)
            data, samplerate  = functions.handle_uploaded_audio_file(uploaded_file)
            sample_frequency=1/samplerate
            fmax=sample_frequency/2
            duration = len(data)/samplerate #DURATION is the length of the generated sample.
            time = np.arange(0,duration, 1/samplerate)
            st.sidebar.markdown('# Original Signal')
            st.sidebar.audio(file_name)
        
    elif option == "Biosignal":
        data,time,samplerate=functions.arrhythima()
    else :
        st.markdown("<h1 style='text-align: center; color:darkcyan;'>Signal Equalizer</h1>", unsafe_allow_html=True)
        functions.generate_sliders(bin_max_frequency_value=10,slidersNum=10)
#----------------------------------------------------------------------Fourier-------------------------------------------------------------------------------------------------------------------------------------------------------
    if not data==[]:
        fft_sig, amplitude,phase,frequencies=functions.Fourier_transform(data,samplerate)
    
#-----------------------------------------------------------------------Vowels---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        if  option=='Vowels':
            freq_axis_list, amplitude_axis_list,bin_max_frequency_value=functions.bins_separation(frequencies, amplitude ,slidersNum=10)
            sliders_data= functions.generate_sliders(bin_max_frequency_value,slidersNum=10)
            mod_amplitude_axis_list,empty= functions.signal_modification(sliders_data,amplitude_axis_list,slidersNum=10)
            phase=phase[:len(mod_amplitude_axis_list):1]
            ifft_file=functions.inverse_fourier(mod_amplitude_axis_list,phase) 
            st.sidebar.markdown('# Modified Signal')
            uploaded_file=ipd.Audio(ifft_file,rate=samplerate/2)
            audio=st.sidebar.write(uploaded_file)
            frequency= frequencies[:len(mod_amplitude_axis_list):1]
            modified_time_axis=np.linspace(0, duration, len(mod_amplitude_axis_list))
            

    #----------------------------------------------------------------------Music-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       

        elif option=='Music' :
            freq_axis_list, amplitude_axis_list,bin_max_frequency_value=functions.bins_separation(frequencies, amplitude ,slidersNum=3)
            sliders_data= functions.generate_sliders(bin_max_frequency_value,slidersNum=3)
            mod_amplitude_axis_list, empty = functions.music_modification(frequencies, amplitude, sliders_data)
            phase=phase[:len(mod_amplitude_axis_list):1]
            ifft_file = functions.inverse_fourier(mod_amplitude_axis_list,phase)
            modified_time_axis = np.linspace(0, duration, len(mod_amplitude_axis_list))
            st.sidebar.markdown('# Modified Signal')
            song = ipd.Audio(ifft_file, rate=samplerate/2)
            st.sidebar.write(song)
            ax = plt.figure(figsize=(10, 8))

    #----------------------------------------------------------------------Biosignal------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif option=='Biosignal':
            freq_axis_list, amplitude_axis_list,bin_max_frequency_value=functions.bins_separation(frequencies, amplitude ,slidersNum=4)
            sliders_data= functions.generate_sliders(bin_max_frequency_value,slidersNum=4)
            mod_amplitude_axis_list,empty= functions.signal_modification(sliders_data,amplitude_axis_list,slidersNum=4)
            phase=phase[:len(mod_amplitude_axis_list):1]
            ifft_file = functions.inverse_fourier(mod_amplitude_axis_list,phase)
            modified_time_axis = np.linspace(0, time, len(mod_amplitude_axis_list))
            

    #-----------------------------------------------------------------------Sine wave-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        elif option=='Sine waves':
            freq_axis_list, amplitude_axis_list,bin_max_frequency_value=functions.bins_separation(frequencies, amplitude ,slidersNum=10)
            sliders_data= functions.generate_sliders(bin_max_frequency_value,slidersNum=10)
            mod_amplitude_axis_list,empty= functions.signal_modification(sliders_data,amplitude_axis_list,slidersNum=10)
            phase=phase[:len(mod_amplitude_axis_list):1]
            ifft_file = functions.inverse_fourier(mod_amplitude_axis_list,phase)
            modified_time_axis = np.linspace(0, duration, len(mod_amplitude_axis_list))
        

    #------------------------------------------------------------------------Static-plotting--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        functions.plot_signal(time,data,frequencies,amplitude) #time-domain representation, This shows us the loudness (amplitude) of sound wave changing with time.




    #------------------------------------------------------------------------Dynamic-Plotting-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        original_time_axis = np.linspace(0, duration, len(data))
        original_df = pd.DataFrame({'time': original_time_axis[::500], 'amplitude': data[:: 500]}, columns=['time', 'amplitude'])
        modified_df=pd.DataFrame({'time': modified_time_axis[::500],'amplitude':mod_amplitude_axis_list[::500]}, columns=['time','amplitude'])
        lines= functions.altair_plot(original_df,modified_df)
        line_plot = st.altair_chart(lines)
        start_btn = st.button('Start')

        if start_btn:
            functions.dynamic_plot(line_plot,original_df,modified_df)

    #---------------------------------------------------------------------------Spectrogram----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        st.sidebar.markdown('## Spectrogram')
        spec1 = st.sidebar.checkbox("Show", key=2)
        if spec1:
            functions.plot_spectrogram(data,fft_sig,samplerate,mod_amplitude_axis_list)






















#     freq_axis_list, amplitude_axis_list,bin_max_frequency_value=functions.bins_separation(sample_frequency, amplitude ,slidersNum=4)

#     sliders_data= functions.generate_sliders(bin_max_frequency_value,slidersNum=4)  

#     mod_amplitude_axis_list,empty= functions.sound_modification(sliders_data,amplitude_axis_list)

#     modified_time_axis=np.linspace(0, duration, len(mod_amplitude_axis_list)) 

#     phase=phase[:len(mod_amplitude_axis_list):1]

#     ifft_file=functions.inverse_fourier(mod_amplitude_axis_list,phase) 

#     frequency= sample_frequency[:len(mod_amplitude_axis_list):1]

#     functions.show_signal(modified_time_axis,ifft_file)
#     functions.show_signal(time,data) #plots wav file data in time domain       
#     functions.plot_spectrogram(data,ifft_file,samplerate,mod_amplitude_axis_list) 















# else :
#     functions.generate_sliders(bin_max_frequency_value=10,slidersNum=10)

    