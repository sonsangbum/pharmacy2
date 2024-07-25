# 실행은 터미널에서 streamlit run pharmacy2.py
#https://github.com/ 에서 URL생성
#https://streamlit.io/ 에 파일업데이트 

import streamlit as st
from PIL import Image   #이미지를 불러올때 사용


#사이드바 화면
st.sidebar.header("로그인")
user_id = st.sidebar.text_input("아이디 입력", value="",max_chars=15)
user_password = st.sidebar.text_input("패스워드 입력", value="",type="password")

if user_id=='phar' and user_password == "1234" :
        
    st.sidebar.header("차트 목록")
    sel_options=["","조제건수","조제건수_이상값","요일별","평균습도","최저습도","일조합","일조율","일사합","강수량","최고기온","최저기온","일교차","평균풍속","최대풍속풍향","최대순간풍속","최대순간풍속방향"]
    user_opt = st.sidebar.selectbox('보고 싶은 차트는? ', sel_options, index=0)
    st.sidebar.write("***선택한 차트는 ", user_opt)

    #메인 화면(오른쪽 화면)
    st.subheader(user_opt,divider='rainbow')
    # st.subheader("메인 화면")
    image_files=["pharmacy.jpg","chart01_조제건수.gif","chart02_조제건수_이상값.gif","chart03_요일vs조제건수.png",
                 "chart04_평균습도(%)vs조제건수.png","chart05_최저습도(%)vs조제건수.png","chart06_일조합(hr)vs조제건수.png",
                 "chart07_일조율(%)vs조제건수.png","chart08_일사합vs조제건수.png","chart09_강수량vs조제건수.png",
                 "chart10_최고기온vs조제건수.png","chart11_최저기온vs조제건수.png","chart12_일교차vs조제건수.png",
                 "chart13_평균풍속vs조제건수.png","chart14_최대풍속vs조제건수.png","chart15_최대풍속풍향vs조제건수.png",
                 "chart16_최대순간풍속vs조제건수.png","chart17_최대순간풍속방향vs조제건수.png"]
    sel_index = sel_options.index(user_opt)
    img_file = image_files[sel_index]
    img_local = Image.open(f'data/{img_file}')
    st.image(img_local, caption=user_opt)


    #메인 화면(오른쪽 화면)
    # st.subheader(user_opt,divider='rainbow')

    #크라우드에서 내가 보내는 프로그램을 인식할 수 있도록
#터미널 창에 실행
#pip list --format=freeze > requirements.txt
#에서 받은 파일에서
#필요한 라이브러리들만 남기고 모두 삭제


