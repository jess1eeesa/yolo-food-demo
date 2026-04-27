import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import json
import os

st.set_page_config(page_title="「一眼瞬間」：深度解構 YOLO 之物件偵測原理", layout="wide")
st.title("「一眼瞬間」：深度解構 YOLO 卷積神經網絡之物件偵測原理")

def load_db():
    # 這裡的名字已經改為 Food_Data.json
    path = "Food_Data.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

food_db = load_db()

@st.cache_resource
def load_model():
    return YOLO('yolov8s-seg.pt')

model = load_model()

st.sidebar.header("控制面板")
source_option = st.sidebar.radio("選擇輸入方式：", ["攝像頭拍照", "上傳照片樣本"])
conf_level = st.sidebar.slider("信心門檻", 0.0, 1.0, 0.35)

img_file = None
if source_option == "攝像頭拍照":
    img_file = st.camera_input("請對準樣本進行採樣")
else:
    img_file = st.file_uploader("上傳影像樣本", type=["jpg", "png", "jpeg"])

if img_file:
    image = Image.open(img_file)
    results = model.predict(source=image, conf=conf_level)
    res = results[0]

    t1, t2, t3, t4 = st.tabs(["感知 (特徵)", "定位 (網格)", "分割 (像素)", "認知 (推理)"])

    with t1:
        edges = cv2.Canny(np.array(image), 100, 200)
        st.image(edges, use_container_width=True)
    with t2:
        st.image(res.plot(labels=True, boxes=True, masks=False), use_container_width=True)
    with t3:
        st.image(res.plot(labels=True, boxes=False, masks=True), use_container_width=True)
    with t4:
        st.subheader("核心原理四：數據推理與科學評估報告")
        if res.boxes:
            for i, box in enumerate(res.boxes):
                label = model.names[int(box.cls[0])]
                conf = box.conf[0]
                info = food_db.get(label, {"name": label, "cal": "待補", "desc": "正在從數據庫擴展特徵匹配..."})
                with st.container(border=True):
                    c1, c2 = st.columns([1, 3])
                    c1.metric(f"目標 {i+1}", info['name'])
                    c2.markdown(f"**識別信心度:** {conf:.2%}")
                    c2.markdown(f"**預估熱量:** {info['cal']} kcal/100g")
                    c2.markdown(f"**分析結果:** {info['desc']}")
        else:
            st.warning("未偵測到樣本。")
