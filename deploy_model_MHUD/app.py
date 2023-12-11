import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prediction import predict

st.title('AMAZON EMPLOYEE ACCESS')
st.header('Employee Features')

col1, col2 = st.columns(2)
with col1:
    resource = st.text_input('RESOURCE')
    mgr_id = st.text_input('MGR_ID')
    role_rollup1 = st.text_input('ROLE_ROLLUP_1')
    role_rollup2 = st.text_input('ROLE_ROLLUP_2')
    role_deptname = st.text_input('ROLE_DEPTNAME')
with col2:
    role_title = st.text_input('ROLE_TITLE')
    role_family_desc = st.text_input('ROLE_FAMILY_DESC')
    role_family = st.text_input('ROLE_FAMILY')
    role_code = st.text_input('ROLE_CODE')
    
st.text('')

if st.button("Predict:"):
    resource = int(resource)
    mgr_id = int(mgr_id)
    role_rollup1 = int(role_rollup1)
    role_rollup2 = int(role_rollup2)
    role_deptname = int(role_deptname)
    role_title = int(role_title)
    role_family_desc = int(role_family_desc)
    role_family = int(role_family)
    role_code = int(role_code)
    result = predict(np.array([[resource, mgr_id, role_rollup1, role_rollup2, role_deptname, role_title, role_family_desc, role_family, role_code]]))
    if result == 0:
        st.title('NO')
    else:
        st.title('YES')
    