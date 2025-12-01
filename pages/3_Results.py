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
st.subheader("3. Kết quả và thông số mô hình được huấn luyện")
import pandas as pd

with st.expander("Các khái niệm"):
    df = pd.DataFrame(
    {
        "Model": ["**YOLOv10-N**", "**YOLOv10-S**", "**YOLOv10-M**", "**YOLOv10-B**", "**YOLOv10-L**", "**YOLOv10-X**", ],
        "Test Size": ["640", "640", "640", "640", "640", "640",],
        "#Params": ["2.3M", "7.2M", "15.4M", "19.1M", "24.4M", "29.5M",],
        "FLOPs": ["6.7G", "21.6G", "59.1G", "92.0G", "120.3G", "160.4G",],
        "APval": ["38.5%", "46.3%", "51.1%", "52.5%", "53.2%", "54.4%",],
        "Latency": ["1.84ms", "2.49ms", "4.74ms", "5.74ms", "7.28ms", "10.70ms",],
    }
)

    st.markdown("**Precision** là tỷ lệ giữa số phát hiện đúng (true positive) và tổng số lần phát hiện (bao gồm cả positive và negative). Nó đo lường độ chính xác của các dự đoán positive và được tính như sau:")
    st.image("precision.jpg")
    st.markdown("**Recall** là tỷ lệ giữa số phát hiện đúng (true positive) và tổng số trường hợp thực tế (ground truth instances). Nó đo lường khả năng của mô hình trong việc tìm ra tất cả các trường hợp liên quan và được tính như sau:")
    st.image("recall.jpg")
    st.markdown("**Intersection over Union (IoU)** là tỷ lệ giữa diện tích giao (intersection area) và diện tích hợp (union area) của hộp dự đoán (predicted bounding box) và hộp thực tế (ground truth bounding box).")
    st.image("iou.jpg")
    st.markdown("**Confidence Score** đại diện cho độ tin cậy của mô hình rằng một đối tượng có mặt trong hộp dự đoán (predicted box), và được tính như sau:")
    st.image("conf.jpg")

    st.table(df)

    st.markdown("**YOLOv10n** : Phiên bản Nano dành cho môi trường có tài nguyên cực kỳ hạn chế.")
    st.markdown("**YOLOv10s** : Phiên bản nhỏ cân bằng tốc độ và độ chính xác.")
    st.markdown("**YOLOv10m** : Phiên bản trung bình dành cho mục đích sử dụng chung.")
    st.markdown("**YOLOv10b** : Phiên bản cân bằng có chiều rộng lớn hơn để có độ chính xác cao hơn.")
    st.markdown("**YOLOv10l** : Phiên bản lớn hơn cho độ chính xác cao hơn với nhưng tăng tài nguyên tính toán.")
    st.markdown("**YOLOv10x** : Phiên bản cực lớn cho độ chính xác và hiệu suất tối đa.")

    st.markdown("**Test Size**: Kích thước của ảnh đầu vào được sử dụng trong quá trình kiểm thử (test). Kích thước ảnh đầu vào ảnh hưởng đến thời gian tính toán và độ chính xác của mô hình. Kích thước càng lớn, mô hình có thể phát hiện được chi tiết nhỏ hơn nhưng cũng yêu cầu nhiều tài nguyên tính toán hơn.")
    st.markdown("**#Param (Số lượng tham số)**: Số lượng tham số huấn luyện (parameters) trong mô hình. Tham số này đại diện cho tất cả các trọng số (weights) mà mô hình cần học để tối ưu hóa trong quá trình huấn luyện. Mô hình có số tham số càng lớn thường có khả năng học phức tạp hơn, nhưng cũng yêu cầu nhiều bộ nhớ và thời gian huấn luyện hơn.")
    st.markdown("**FLOPs (Floating Point Operations)**: FLOPs là số phép toán dấu chấm động mà mô hình thực hiện để xử lý một ảnh. Nó thể hiện mức độ phức tạp tính toán của mô hình. Một mô hình có FLOPs thấp sẽ nhanh hơn trong việc thực thi và tiết kiệm tài nguyên tính toán, nhưng có thể giảm độ chính xác.")
    st.markdown("**APval (Average Precision at validation)** là chỉ số chính xác trên bộ dữ liệu kiểm tra (validation dataset). Chỉ số này đánh giá khả năng của mô hình trong việc phát hiện các đối tượng đúng với nhiều điều kiện như độ phân giải ảnh, độ sáng, góc nhìn, v.v. Giá trị AP cao đồng nghĩa với khả năng phát hiện đối tượng tốt hơn.")
    st.markdown("**Latency (Độ trễ)**: Độ trễ là thời gian mà mô hình cần để xử lý một ảnh từ khi nhận vào cho đến khi đưa ra kết quả (thường tính bằng mili giây). Độ trễ càng thấp, mô hình càng nhanh và phù hợp với các ứng dụng thời gian thực.")
    st.markdown("**Mean Average Precision (mAP)** là một chỉ số được sử dụng để đánh giá hiệu quả của các thuật toán phát hiện đối tượng trong việc nhận diện và xác định vị trí của các đối tượng trong hình ảnh. Nó xem xét cả độ chính xác (precision) và độ thu hồi (recall) trên các danh mục khác nhau. Bằng cách tính toán độ chính xác trung bình (Average Precision - AP) cho từng danh mục và lấy giá trị trung bình, mAP cung cấp đánh giá tổng thể về hiệu suất của thuật toán.")
  

with st.expander("Các thông số:"):
    st.image("label.jpg")
    st.image("labels.jpg")
    st.image("1.png")
    st.image("6.png")
    st.image("7.png")
    st.markdown("Trục hoành của đồ thị là số lượng epochs. Một epoch là một vòng lặp hoàn chỉnh của bộ dữ liệu huấn luyện qua mạng nơ-ron trong quá trình huấn luyện. Thêm nhiều epochs thường dẫn đến việc huấn luyện mô hình tốt hơn, nhưng nếu quá mức có thể dẫn đến hiện tượng overfitting.")
    st.markdown("Trục tung trong đại diện cho giá trị loss. Loss là một thước đo đánh giá hiệu suất của mô hình trong quá trình huấn luyện. Giá trị loss thấp hơn chỉ ra hiệu suất mô hình tốt hơn, trong khi giá trị cao hơn chỉ ra hiệu suất kém. Mục tiêu trong quá trình huấn luyện là giảm thiểu giá trị loss.")
    st.markdown("**box_loss**: Giá trị loss của Bounding Box, đo lỗi trong tọa độ và kích thước của Bounding Box so với ground truth. box_loss thấp có nghĩa là các Bounding Box dự đoán chính xác hơn.")
    st.markdown("**cls_loss**: Đây là loss phân loại class, đo lường lỗi trong xác suất phân loại dự đoán cho mỗi đối tượng trong hình ảnh so với ground truth. Một giá trị cls_loss thấp hơn có nghĩa là mô hình dự đoán lớp của các đối tượng chính xác hơn.")
    st.markdown("**dfl_loss**: Loss của lớp convolution biến dạng (deformable convolution layer), một bổ sung mới vào kiến trúc YOLO kể từ YOLOv8. Loss này đo lường lỗi trong các lớp convolution biến dạng, được thiết kế để cải thiện khả năng của mô hình trong việc phát hiện các đối tượng có nhiều tỷ lệ kích thước và tỉ lệ khung hình khác nhau. Một giá trị dfl_loss thấp hơn chỉ ra rằng mô hình xử lý tốt hơn sự biến dạng của đối tượng và các thay đổi về hình dạng.")
    st.image("10.jpg")
    st.markdown("Đường cong precision và recall của YOLOv10n có giá trị mAP trung bình là 61.1%, với lớp **car** có giá trị mAP cao nhất là 0.809%. Trong khi đó, lớp **truck** có giá trị mAP cao thứ hai là 77.2%. ")
    st.markdown("Trong mô hình YOLOv10s, giá trị mAP tổng thể của tất cả các lớp là 67.7%, với lớp **car** có giá trị mAP cao nhất là 0.842%, tiếp theo là lớp **truck** với giá trị mAP cao thứ hai là 75%.")
    st.markdown("Các kết quả này cho thấy hiệu suất của mô hình phát hiện đối tượng với việc tăng cường dữ liệu xoay ngang (horizontal flip). Giá trị mAP trung bình và mAP của lớp cao nhất cung cấp thông tin về khả năng của mô hình trong việc phát hiện các đối tượng trong hình ảnh. Các kết quả cao hơn chỉ ra rằng mô hình hoạt động tốt hơn trong nhiệm vụ phát hiện đối tượng, và các lớp **car** và **truck** liên tục có mAP cao nhất trong các lớp của mô hình. ")
    st.image("102.jpg")



with st.expander("So sánh:"):
    if st.checkbox("Video giao thông"):
        cols = st.columns(2)
        with cols[0]: st.video("videos/N-D1/traffic.mp4")
        with cols[1]: st.video("videos/S-D1/traffic.mp4")
        st.caption("YOLO nano - Dataset               |              YOLO small - Dataset")
        if st.button("Play all videos", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )
            
    if st.checkbox("Video giao thông 2"):
        cols2 = st.columns(2)
        with cols2[0]: st.video("videos/N-D1/traffic2.mp4")
        with cols2[1]: st.video("videos/S-D1/traffic2.mp4")
        st.caption("YOLO nano - Dataset               |              YOLO small - Dataset")
        if st.button("Play all videos 2", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )

    if st.checkbox("Video giao thông 3"):
        cols3 = st.columns(2)
        with cols3[0]: st.video("videos/N-D1/traffic3.mp4")
        with cols3[1]: st.video("videos/S-D1/traffic3.mp4")
        st.caption("YOLO nano - Dataset               |              YOLO small - Dataset")
        if st.button("Play all videos 3", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )

    if st.checkbox("Video giao thông 4"):
        cols4 = st.columns(2)
        with cols4[0]: st.video("videos/N-D1/traffic4.mp4")
        with cols4[1]: st.video("videos/S-D1/traffic4.mp4")
        st.caption("YOLO nano - Dataset               |              YOLO small - Dataset")
        if st.button("Play all videos 4", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )

    if st.checkbox("Video giao thông 5"):
        cols5 = st.columns(2)
        with cols5[0]: st.video("videos/N-D1/traffic5.mp4")
        with cols5[1]: st.video("videos/S-D1/traffic5.mp4")
        st.caption("YOLO nano - Dataset               |              YOLO small - Dataset")
        if st.button("Play all videos 5", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )

    if st.checkbox("Video giao thông 6"):
        cols6 = st.columns(2)
        with cols6[0]: st.video("videos/N-D1/traffic6.mp4")
        with cols6[1]: st.video("videos/S-D1/traffic6.mp4")
        st.caption("YOLO nano - Dataset               |              YOLO small - Dataset")
        if st.button("Play all videos 6", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )

        
    if st.checkbox("Video giao thông 10"):
        cols9 = st.columns(2)
        with cols9[0]: st.video("videos/N-D2/traffic2.mp4")
        with cols9[1]: st.video("videos/N-D6/traffic5.mp4")
        st.caption("YOLO nano - Dataset 1               |              YOLO nano - Dataset 2")
        if st.button("Play all videos 10", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )
            
            
    if st.checkbox("Video giao thông 11"):
        cols9 = st.columns(2)
        with cols9[0]: st.video("videos/N-D2/traffic5.mp4")
        with cols9[1]: st.video("videos/N-D6/traffic6.mp4")
        st.caption("YOLO nano - Dataset 1               |              YOLO nano - Dataset 2")
        if st.button("Play all videos 11", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )
            
            
    if st.checkbox("Video giao thông 12"):
        cols9 = st.columns(2)
        with cols9[0]: st.video("videos/N-D2/traffic.mp4")
        with cols9[1]: st.video("videos/N-D6/traffic2.mp4")
        st.caption("YOLO nano - Dataset 1               |              YOLO nano - Dataset 2")
        if st.button("Play all videos 12", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )
                   
            
            
    if st.checkbox("Video giao thông 13"):
        cols9 = st.columns(2)
        with cols9[0]: st.video("videos/N-D2/traffic3.mp4")
        with cols9[1]: st.video("videos/N-D6/traffic3.mp4")
        st.caption("YOLO nano - Dataset 1               |              YOLO nano - Dataset 2")
        if st.button("Play all videos 13", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )
                   
            
            
    if st.checkbox("Video giao thông 14"):
        cols9 = st.columns(2)
        with cols9[0]: st.video("videos/N-D2/traffic4.mp4")
        with cols9[1]: st.video("videos/N-D6/traffic4.mp4")
        st.caption("YOLO nano - Dataset 1               |              YOLO nano - Dataset 2")
        if st.button("Play all videos 14", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )
                   
            
    if st.checkbox("Video giao thông 15"):
        cols9 = st.columns(2)
        with cols9[0]: st.video("videos/N-D2/traffic8.mp4")
        with cols9[1]: st.video("videos/N-D6/traffic1.mp4")
        st.caption("YOLO nano - Dataset 1               |              YOLO nano - Dataset 2")
        if st.button("Play all videos 15", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )
                   
            
            
    if st.checkbox("Video giao thông 16"):
        cols9 = st.columns(2)
        with cols9[0]: st.video("videos/N-D2/traffic7.mp4")
        with cols9[1]: st.video("videos/N-D6/traffic7.mp4")
        st.caption("YOLO nano - Dataset 1               |              YOLO nano - Dataset 2")
        if st.button("Play all videos 16", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )
                   
            
    if st.checkbox("Video giao thông 17"):
        cols9 = st.columns(2)
        with cols9[0]: st.video("videos/N-D2/traffic6.mp4")
        with cols9[1]: st.video("videos/N-D6/traffic9.mp4")
        st.caption("YOLO nano - Dataset 1               |              YOLO nano - Dataset 2")
        if st.button("Play all videos 17", use_container_width=True):
            st.components.v1.html(
                """<script>
                let videos = parent.document.querySelectorAll("video");
                videos.forEach(v => {
                    v.play();
                })
                </script>""", 
                width=0, height=0
            )
                   
            
                                       
            
  
                       
            
            
  
            
            
  
            

