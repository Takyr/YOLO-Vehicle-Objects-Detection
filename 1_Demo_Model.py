from track import *
import tempfile
import cv2
import torch
import streamlit as st
import os

# main page
temperature1 = "TS. Nguyễn Thành Sơn"
temperature2 = "Trương Gia Vỷ  MSSV: 20133115"

st.write(f"GVHD: :blue[{temperature1}]")
st.write(f"SVTH: :blue[{temperature2}]")
st.title(":blue[Nhận diện các phương tiện đang di chuyển trên đường qua camera giao thông ] ")
st.subheader("1. Trình bày mô hình được huấn luyện")






if __name__ == '__main__':
   
    # upload video
    video_file_buffer = st.sidebar.file_uploader("Tải video", type=['mp4', 'mov', 'avi'])

    if video_file_buffer:
        st.sidebar.text('Video đầu vào')
        st.sidebar.video(video_file_buffer)
        # save video from streamlit into "videos" folder for future detect
        with open(os.path.join('videos', video_file_buffer.name), 'wb') as f:
            f.write(video_file_buffer.getbuffer())

    st.sidebar.markdown('---')
    st.sidebar.title('Cài đặt')
    # custom class
    custom_class = st.sidebar.checkbox('Các class phương tiện')
    assigned_class_id = [0, 1, 2, 3]
    names = ['xe hơi', 'mô tô', 'xe tải', 'xe buýt']

    if custom_class:
        assigned_class_id = []
        assigned_class = st.sidebar.multiselect('Chọn các class phương tiện', list(names))
        for each in assigned_class:
            assigned_class_id.append(names.index(each))
    
    # st.write(assigned_class_id)

    # setting hyperparameter
    confidence = st.sidebar.slider('Độ tin cậy', min_value=0.0, max_value=1.0, value=0.5)
    line = st.sidebar.number_input('Vị trí đường kẻ', min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    st.sidebar.markdown('---')

    
    status = st.empty()
    stframe = st.empty()
    if video_file_buffer is None:
        status.markdown('<font size= "4"> **Status:** Đang đợi... </font>', unsafe_allow_html=True)
    else:
        status.markdown('<font size= "4"> **Status:** Sẵn sàng </font>', unsafe_allow_html=True)

    car, bus, truck, motor = st.columns(4)
    with car:
        st.markdown('**Xe hơi**')
        car_text = st.markdown('__')
    
    with bus:
        st.markdown('**Xe bus**')
        bus_text = st.markdown('__')

    with truck:
        st.markdown('**Xe tải**')
        truck_text = st.markdown('__')
    
    with motor:
        st.markdown('**Mô tô**')
        motor_text = st.markdown('__')

    fps, _,  _, _  = st.columns(4)
    with fps:
        st.markdown('**FPS**')
        fps_text = st.markdown('__')


    track_button = st.sidebar.button('BẮT ĐẦU')
    # reset_button = st.sidebar.button('RESET ID')
    if track_button:
        # reset ID and count from 0
        reset()
        opt = parse_opt()
        opt.conf_thres = confidence
        opt.source = f'videos/{video_file_buffer.name}'

        status.markdown('<font size= "4"> **Status:** Đang chạy... </font>', unsafe_allow_html=True)
        with torch.no_grad():
            detect(opt, stframe, car_text, bus_text, truck_text, motor_text, line, fps_text, assigned_class_id)
        status.markdown('<font size= "4"> **Status:** Hoàn thành ! </font>', unsafe_allow_html=True)
        # end_noti = st.markdown('<center style="color: blue"> FINISH </center>',  unsafe_allow_html=True)

    # if reset_button:
        # reset()
    #     st.markdown('<h3 style="color: blue"> Reseted ID </h3>', unsafe_allow_html=True)
