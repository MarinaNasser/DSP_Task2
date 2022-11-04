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

def show_signal(x_axis,y_axis):
    SignalFigure, SignalAxis = plt.subplots(1, 1)
    SignalAxis.plot(x_axis,y_axis)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    st.plotly_chart(SignalFigure,use_container_width=True)
