import streamlit as st

def Change_play_State():
    st.session_state['play_state']=  not st.session_state['play_state']