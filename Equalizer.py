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
with open('Equalizer.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)



#------------------------------------------------------------------Upload_file----------------------------------------------------------------------------------------------------------------------------------------------

option = st.selectbox("Pick your sample!", options=["Take your pick", "Music", "Biosignal", "Sine waves", "Vowels"])

uploaded_file = st.sidebar.file_uploader("uploader",key="uploaded_file",label_visibility="hidden")


if uploaded_file is None :
       functions.generate_sliders(bin_max_frequency_value=10 , slidersNum=10 )


if uploaded_file is not None:
    file_name=uploaded_file.name
    ext = os.path.splitext(file_name)[1][1:]
  
    #------------------------------------------------------------------csv----------------------------------------------------------------------------------------------------------------------------------------------
    if ext=='csv':
        df = pd.read_csv(uploaded_file)
        list_of_columns=df.columns
        time = df[list_of_columns[0]].to_numpy()
        data = df[list_of_columns[1]].to_numpy()
        max_freq=functions.getFMax(time,data)
        samplerate=2*max_freq
        duration = len(time) 
        

    #------------------------------------------------------------------wav----------------------------------------------------------------------------------------------------------------------------------------------
    elif ext=='wav':
         # functions.Audio_player(uploaded_file)
        data, samplerate  = functions.handle_uploaded_audio_file(uploaded_file)
        sample_frequency=1/samplerate
        fmax=sample_frequency/2
        duration = len(data)/samplerate #DURATION is the length of the generated sample.
        time = np.arange(0,duration, 1/samplerate)
        st.sidebar.markdown('# Original Signal')
        st.sidebar.audio(file_name)


#----------------------------------------------------------------------------------------------------------------------------------------------------biosignal-------------------------------------------------


    if option == "Biosignal":
        data,time,samplerate=functions.arrhythima()
        fft_sig, amplitude, phase, frequency = functions.Fourier_transform(
                data, samplerate)
        bin_max_frequency_value = functions.bins_separation(frequency, amplitude, slidersNum=4)        
        sliders_data = functions.generate_sliders(bin_max_frequency_value,slidersNum=4)
        modified_amplitude, empty = functions.signal_modification(
             sliders_data, amplitude, slidersNum=4)
        ifft_file = functions.inverse_fourier(modified_amplitude, phase)
        functions.plot_signal(time,ifft_file,frequency,modified_amplitude)

            
#----------------------------------------------------------------------------------------------------------------------------------------------------music-------------------------------------------------

    elif option == "Music" :

        sliders_data = functions.music_generate_sliders()
        fft_sig, amplitude, phase, frequency = functions.Fourier_transform(
                data, samplerate)
        modified_amplitude, empty = functions.music_modification(
        frequency, amplitude, sliders_data)
        modified_time_axis = np.linspace(
                0, duration, len(modified_amplitude))
        ifft_file = functions.inverse_fourier(modified_amplitude, phase)
        song = ipd.Audio(ifft_file, rate=samplerate/2)
        empty.write(song)
        ax = plt.figure(figsize=(10, 8))

#------------------------------------------------------------------------------------------------------------------------------------------------------sine wave-----------------------------------------------
    # elif option == "Sine waves" :

#-----------------------------------------------------------------------------------------------------------------------------------------------------vowels------------------------------------------------  
    # elif option == "Vowels" : 

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
   




        # fft_sig, amplitude,phase,frequencies=functions.Fourier_transform(data,samplerate)
        # freq_axis_list, amplitude_axis_list,bin_max_frequency_value=functions.bins_separation(frequencies, amplitude ,slidersNum=10)
        # sliders_data= functions.generate_sliders(bin_max_frequency_value,slidersNum=10)
        # mod_amplitude_axis_list,empty= functions.signal_modification(sliders_data,amplitude_axis_list,slidersNum=10)
        # phase=phase[:len(mod_amplitude_axis_list):1]
        # ifft_file=functions.inverse_fourier(mod_amplitude_axis_list,phase) 

        # if option=='Music' or option=='Vowels':
        #     # modified_time_axis=np.linspace(0, duration, len(mod_amplitude_axis_list))
        #     # st.markdown('# Modified Signal')
        #     uploaded_file=ipd.Audio(ifft_file,rate=samplerate/2)
        #     audio=empty.write(uploaded_file)
        #     frequency= frequencies[:len(mod_amplitude_axis_list):1]
            # original_time_axis = np.linspace(0, duration, len(data))

            # fft_sig, amplitude, phase, frequency = functions.Fourier_transform(
            #     data, samplerate)
            # sliders_data = functions.music_generate_sliders()
            # modified_amplitude, empty = functions.music_modification(
            #     frequency, amplitude, sliders_data)
            # modified_time_axis = np.linspace(
            #     0, duration, len(modified_amplitude))
            # ifft_file = functions.inverse_fourier(modified_amplitude, phase)
            # song = ipd.Audio(ifft_file, rate=samplerate/2)
            # empty.write(song)
            # ax = plt.figure(figsize=(10, 8))


        
#-------------------------------------------------------------------------------plotting-------------------------------------------------------------------------------------------------------------------


        # functions.plot_signal(time,data,frequencies,amplitude) #time-domain representation, This shows us the loudness (amplitude) of sound wave changing with time.
        # original_time_axis = np.linspace(0, duration, len(data))
        # original_df = pd.DataFrame({'time': original_time_axis[::500], 'amplitude': data[:: 500]}, columns=[
        #     'time', 'amplitude'])
        # modified_time_axis = np.linspace(0, duration, len(mod_amplitude_axis_list))
        # modified_df=pd.DataFrame({'time': modified_time_axis[::500],'amplitude':mod_amplitude_axis_list[::500]}, columns=['time','amplitude'])
        # lines= functions.altair_plot(original_df,modified_df)
        # line_plot = st.altair_chart(lines)
        # start_btn = st.button('Start')

        # if start_btn:
        #     functions.dynamic_plot(line_plot,original_df,modified_df)
        
        # functions.plot_spectrogram(data,fft_sig,samplerate,mod_amplitude_axis_list)




         