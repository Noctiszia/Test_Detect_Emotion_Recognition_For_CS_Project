#!/bin/bash

# ติดตั้ง Streamlit และ Streamlit Webrtc ในระบบของ Heroku
pip install streamlit streamlit-webrtc

# ตั้งค่า Streamlit
echo "\
[general]\n\
\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

