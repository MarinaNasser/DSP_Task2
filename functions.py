import streamlit as st
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import matplotlib.pyplot as plt


def Change_play_State():
    st.session_state['play_state']=  not st.session_state['play_state']


# Read the Audiofile
def handle_uploaded_audio_file(uploaded_file):
    a = AudioSegment.from_wav(uploaded_file)
    samples = a.get_array_of_samples()
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples.typecode).max
    
    return fp_arr,  48000 #22050 



def showthesignal(duration , samplerate, data):
    time = np.arange(0,duration,1/samplerate)
    # Plotting the Graph using Matplotlib
    SignalFigure, SignalAxis = plt.subplots(1, 1)
    SignalAxis.plot(time,data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    st.plotly_chart(SignalFigure,use_container_width=True)
