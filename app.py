# app.py

import streamlit as st
import joblib
import numpy as np

# 모델 로드 (pkl 파일이 같은 폴더에 있어야 함)
model = joblib.load('fraud_model_core.pkl')

st.title("💳 사기 거래 예측기")

# 입력창
V14 = st.number_input("V14", value=0.0)
V10 = st.number_input("V10", value=0.0)
V17 = st.number_input("V17", value=0.0)
V11 = st.number_input("V11", value=0.0)
V4 = st.number_input("V4", value=0.0)
amount = st.number_input("scaled_amount", value=0.3)
hour = st.slider("Hour", 0, 23, 14)
is_night = st.selectbox("is_night", [0, 1])

if st.button("예측하기"):
    input_array = np.array([[V14, V10, V17, V11, V4, amount, hour, is_night]])
    prob = model.predict_proba(input_array)[0][1]

    st.metric("사기 거래 확률", f"{round(prob * 100, 2)}%")

    # 등급 & 메시지
    if prob <= 0.2:
        level, msg = "✅ 안전", "정상적인 거래로 판단됩니다."
    elif prob <= 0.6:
        level, msg = "⚠️ 주의", "약간의 사기 징후가 있습니다."
    elif prob <= 0.85:
        level, msg = "❗ 경고", "사기 가능성이 높습니다. 검토를 권장합니다."
    else:
        level, msg = "🚨 위험", "사기 가능성이 매우 높습니다. 즉시 차단을 권장합니다."

    st.write(f"등급: {level}")
    st.info(msg)