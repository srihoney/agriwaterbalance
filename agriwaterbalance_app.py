import streamlit as st

st.set_page_config(page_title="Test App", layout="wide")

with st.sidebar:
    st.header("Header 1")
    st.write("Content 1")
    
    st.header("Header 2")
    st.write("Content 2")
    
    st.header("Header 3")
    st.write("Content 3")
    
    st.header("Header 4")
    st.write("Content 4")
    
st.write("Main app content here.")
