import streamlit as st
import pandas as pd
from io import StringIO
import pickle

def preprocess_and_predict(model, test_df):
    features_to_keep = [
        'ps_car_13', 'ps_ind_03', 'ps_ind_05_cat', 'ps_reg_02', 'ps_ind_17_bin', 'ps_ind_15',
        'ps_reg_01', 'ps_car_14', 'ps_car_01_cat', 'ps_ind_01', 'ps_car_07_cat', 'ps_car_11_cat',
        'ps_calc_03', 'ps_ind_07_bin', 'ps_car_09_cat', 'ps_ind_02_cat', 'ps_calc_10', 'ps_car_15',
        'ps_calc_14', 'ps_car_06_cat', 'ps_ind_06_bin', 'ps_calc_11', 'ps_ind_16_bin', 'ps_calc_01',
        'ps_car_12'
    ]

    if 'id' not in test_df.columns:
        st.error("CSV ไม่มีคอลัมน์ 'id'")
        return None
    
    test_id = test_df['id']
    X_test_processed = pd.DataFrame()
    for col in features_to_keep:
        if col in test_df.columns:
            X_test_processed[col] = test_df[col]
        else:
            X_test_processed[col] = -1

    X_test_processed = X_test_processed.fillna(-1)
    X_test_processed[X_test_processed < -1] = -1

    st.info(f"ทำการทำนายข้อมูลจำนวน {X_test_processed.shape[0]} แถว")

    try:
        prediction_proba = model.predict_proba(X_test_processed)[:, 1]
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่าง predict: {e}")
        return None
    
    submission_df = pd.DataFrame({
        'id': test_id,
        'target': prediction_proba
    })
    
    return submission_df

st.set_page_config(
    page_title="Kaggle Submission",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("Porto Seguro Prediction (Kaggle)")
st.markdown("---")

st.header("อัปโหลดไฟล์ทดสอบ (test.csv)")
st.info("ไฟล์ที่อัปโหลดควรมีคอลัมน์ 'id' และคอลัมน์ฟีเจอร์เหมือนตอน train")

uploaded_file = st.file_uploader("เลือกไฟล์ CSV สำหรับทำนาย (เช่น test.csv)", type=['csv'])
submission_df = None

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file, dtype='float32')
        st.success(f"ไฟล์ {uploaded_file.name} ถูกโหลดแล้ว ({test_data.shape[0]} แถว, {test_data.shape[1]} คอลัมน์)")
        st.dataframe(test_data.head())

        if st.button("ประมวลผลและสร้างไฟล์ Submission"):
            with st.spinner('กำลังประมวลผลและทำนาย...'):
                submission_df = preprocess_and_predict(model, test_data)

            if submission_df is not None:
                st.subheader("ผลการทำนายพร้อมดาวน์โหลด")
                st.dataframe(submission_df.head())

                csv_buffer = StringIO()
                submission_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                downloaded = st.download_button(
                    label="ดาวน์โหลด Submission CSV",
                    data=csv_data,
                    file_name='submission.csv',
                    mime='text/csv'
                )
                
                if downloaded:
                    st.cache_data.clear()
                    st.cache_resource.clear()
            else:
                st.error("ไม่สามารถสร้างไฟล์ Submission ได้เนื่องจากเกิดข้อผิดพลาด")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์หรือประมวลผล: {e}")
