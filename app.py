import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 모델 불러오기 ──────────────────────────────
model_자립률    = joblib.load('model_자립률.pkl')
model_등급      = joblib.load('model_등급.pkl')
model_1차에너지 = joblib.load('model_1차에너지.pkl')
le_등급         = joblib.load('le_등급.pkl')
le_dict         = joblib.load('le_dict.pkl')
feature_cols    = joblib.load('feature_columns.pkl')
df_data         = pd.read_csv('df_model_cleaned.csv')

# ── 페이지 설정 ────────────────────────────────
st.set_page_config(
    page_title="제로에너지 건축물 사전 예측",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 제로에너지 건축물 인증 사전 예측")
st.caption("설계 전 대략적인 에너지자립률과 인증 등급을 예측해드려요!")
st.divider()

# ── 입력 폼 ────────────────────────────────────
st.subheader("📥 건물 정보 입력")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📍 기본 정보**")
    지역 = st.selectbox("지역",
        ['강원','경기','경남','경북','광주','대구',
         '대전','부산','서울','세종','울산','인천',
         '전남','전북','제주','충남','충북'])

    건물용도 = st.selectbox("건물 용도",
        sorted(le_dict['건물용도'].classes_.tolist()))

    연면적 = st.number_input("연면적 (m²)",
        min_value=0.0, value=3000.0, step=100.0)

    창면적비 = st.number_input("창면적비 (%)",
        min_value=0.0, max_value=100.0, value=25.0, step=1.0)

with col2:
    st.markdown("**🌡️ 설비 정보**")
    난방방식 = st.selectbox("난방 방식",
        ['히트펌프', '지역난방', '보일러', '기타'])

    냉방방식 = st.selectbox("냉방 방식",
        ['압축식', '흡수식', '냉방없음', '기타'])

    지열여부 = st.radio("지열 설치",
        ['없음', '있음'], horizontal=True)

    열병합여부 = st.radio("열병합 설치",
        ['없음', '있음'], horizontal=True)

with col3:
    st.markdown("**☀️ 태양광 정보**")
    태양광타입 = st.radio("태양광 설치 타입",
        ['후면통풍형', '밀착형'], horizontal=True)

    태양광면적_입력 = st.number_input("태양광 면적 (m²)",
        min_value=0.0, value=300.0, step=10.0)

    효율입력여부 = st.checkbox("태양광 효율 직접 입력할게요")
    if 효율입력여부:
        태양광효율 = st.number_input("태양광 효율 (%)",
            min_value=0.0, max_value=100.0, value=20.0, step=0.5)
        # 효율 보정: 입력면적 / 12 * 효율
        보정면적 = 태양광면적_입력 / 12 * 태양광효율
        st.caption(f"💡 보정 면적: {보정면적:.1f} m²")
    else:
        보정면적 = 태양광면적_입력

    # 밀착형이면 후면통풍형으로 환산
    if 태양광타입 == '후면통풍형':
        태양광_후면 = 보정면적
        태양광_밀착 = 보정면적 * (0.12 / 0.112)
    else:
        태양광_밀착 = 보정면적
        태양광_후면 = 보정면적 * (0.112 / 0.12)

st.divider()

# ── 예측 버튼 ──────────────────────────────────
if st.button("🔍 예측하기", type="primary", use_container_width=True):

    # 입력값 정리
    건물용도구분  = 0 if '주거' in 건물용도 and '이외' not in 건물용도 else 1

    # 지역 인코딩
    if 지역 in le_dict['지역'].classes_:
        지역_코드 = le_dict['지역'].transform([지역])[0]
    else:
        지역_코드 = le_dict['지역'].transform(['unknown'])[0]

    # 건물용도 인코딩
    if 건물용도 in le_dict['건물용도'].classes_:
        건물용도_코드 = le_dict['건물용도'].transform([건물용도])[0]
    else:
        건물용도_코드 = le_dict['건물용도'].transform(['unknown'])[0]

    난방방식_코드 = {'히트펌프':0,'지역난방':1,'보일러':2,'기타':3}[난방방식]
    냉방방식_코드 = {'압축식':0,'흡수식':1,'냉방없음':2,'기타':3}[냉방방식]

    # 태양광비율
    태양광비율 = 태양광_후면 / 연면적 if 연면적 > 0 else 0

    input_data = pd.DataFrame([{
        '지역'       : 지역_코드,
        '건물용도'   : 건물용도_코드,
        '건물용도구분': 건물용도구분,
        '연면적'     : 연면적,
        '창면적비'   : 창면적비,
        '난방방식'   : 난방방식_코드,
        '냉방방식'   : 냉방방식_코드,
        '태양광용량'  : 0,
        '태양광_후면' : 태양광_후면,
        '태양광_밀착' : 태양광_밀착,
        '지열여부'   : 1 if 지열여부 == '있음' else 0,
        '열병합여부'  : 1 if 열병합여부 == '있음' else 0,
        '태양광비율'  : 태양광비율,
    }])

    input_data = input_data[feature_cols]

    # 예측
    pred_자립률    = model_자립률.predict(input_data)[0]
    pred_1차에너지 = model_1차에너지.predict(input_data)[0]
    pred_등급_num  = model_등급.predict(input_data)[0]
    pred_등급      = le_등급.inverse_transform([pred_등급_num])[0]

    # ── 결과 표시 ──────────────────────────────
    st.subheader("📊 예측 결과")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="⚡ 에너지자립률",
                  value=f"{pred_자립률:.1f}%")

    with col2:
        st.metric(label="🔋 1차에너지소요량",
                  value=f"{pred_1차에너지:.1f} kWh/㎡·년",
                  help="⚠️ 참고용 예측값이에요, 오차가 클 수 있어요!")

    with col3:
        등급_이모지 = {
            '+':'🥇','1':'🥈','2':'🥉',
            '3':'🏅','4':'🎖️','5':'📋','인증불가':'❌'
        }
        st.metric(
            label="🏢 예측 등급",
            value=f"{등급_이모지.get(pred_등급,'')} {pred_등급}등급"
                  if pred_등급 != '인증불가' else "❌ 인증불가"
        )

    # 안내 메시지
    if pred_등급 == '인증불가':
        st.error("⚠️ 현재 조건으로는 인증이 어려워요! 태양광을 늘려보세요!")
    elif pred_등급 in ['+', '1', '2']:
        st.success("🎉 우수한 등급이 예상돼요!")
    elif pred_등급 in ['3', '4']:
        st.info("👍 양호한 등급이 예상돼요!")
    else:
        st.warning("💡 태양광을 조금 더 늘리면 더 높은 등급을 받을 수 있어요!")

    st.divider()

    # ── 그래프 ─────────────────────────────────
    st.subheader("📈 분석 그래프")

    tab1, tab2, tab3 = st.tabs([
        "☀️ 태양광 면적별 자립률 변화",
        "📊 전체 데이터 분포",
        "🎯 등급 기준 비교"
    ])

    # 그래프 1: 태양광 면적 변화에 따른 자립률
    with tab1:
        면적범위 = np.linspace(0, max(태양광면적_입력 * 3, 100), 50)
        자립률예측 = []
        등급예측 = []

        for 면적 in 면적범위:
            if 효율입력여부:
                보정 = 면적 / 12 * 태양광효율
            else:
                보정 = 면적

            if 태양광타입 == '후면통풍형':
                후면 = 보정
                밀착 = 보정 * (0.12/0.112)
            else:
                밀착 = 보정
                후면 = 보정 * (0.112/0.12)

            비율 = 후면 / 연면적 if 연면적 > 0 else 0

            tmp = pd.DataFrame([{
                '지역': 지역_코드, '건물용도': 건물용도_코드,
                '건물용도구분': 건물용도구분, '연면적': 연면적,
                '창면적비': 창면적비, '난방방식': 난방방식_코드,
                '냉방방식': 냉방방식_코드, '태양광용량': 0,
                '태양광_후면': 후면, '태양광_밀착': 밀착,
                '지열여부': 1 if 지열여부 == '있음' else 0,
                '열병합여부': 1 if 열병합여부 == '있음' else 0,
                '태양광비율': 비율,
            }])[feature_cols]

            자립률예측.append(model_자립률.predict(tmp)[0])
            등급num = model_등급.predict(tmp)[0]
            등급예측.append(le_등급.inverse_transform([등급num])[0])

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(면적범위, 자립률예측, color='steelblue', linewidth=2.5)
        ax1.axvline(x=태양광면적_입력, color='red', linestyle='--',
                    linewidth=1.5, label=f'현재 입력값 ({태양광면적_입력:.0f}㎡)')
        ax1.axhline(y=pred_자립률, color='orange', linestyle='--',
                    alpha=0.7, linewidth=1.5)

        # 등급 기준선
        for 기준, 라벨, 색 in [
            (20,'5등급','#aed6f1'),
            (40,'4등급','#a9dfbf'),
            (60,'3등급','#f9e79f'),
            (80,'2등급','#f0b27a'),
            (100,'1등급','#ec7063'),
            (120,'+등급','#c39bd3')
        ]:
            ax1.axhline(y=기준, color=색, linestyle=':', alpha=0.9, linewidth=1.5)
            ax1.text(면적범위[-1]*0.98, 기준+1, 라벨,
                    fontsize=8, ha='right', color='gray')

        ax1.set_xlabel("태양광 면적 (m²)", fontsize=11)
        ax1.set_ylabel("에너지자립률 (%)", fontsize=11)
        ax1.set_title("태양광 면적에 따른 에너지자립률 변화", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

    # 그래프 2: 전체 데이터 분포
    with tab2:
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4))

        ax2a.hist(df_data['에너지자립률'], bins=50,
                  color='steelblue', alpha=0.7, edgecolor='white')
        ax2a.axvline(x=pred_자립률, color='red', linewidth=2,
                     label=f'현재 예측값 ({pred_자립률:.1f}%)')
        percentile = (df_data['에너지자립률'] <= pred_자립률).mean() * 100
        ax2a.set_xlabel("에너지자립률 (%)", fontsize=11)
        ax2a.set_ylabel("건물 수", fontsize=11)
        ax2a.set_title(f"전체 데이터 자립률 분포\n(상위 {100-percentile:.1f}% 수준)", fontsize=11)
        ax2a.legend()
        ax2a.grid(alpha=0.3)

        ax2b.hist(df_data['1차에너지소요량'].clip(-100, 400),
                  bins=50, color='darkorange', alpha=0.7, edgecolor='white')
        ax2b.axvline(x=pred_1차에너지, color='red', linewidth=2,
                     label=f'현재 예측값 ({pred_1차에너지:.1f})')
        ax2b.set_xlabel("1차에너지소요량 (kWh/㎡·년)", fontsize=11)
        ax2b.set_ylabel("건물 수", fontsize=11)
        ax2b.set_title("전체 데이터 1차에너지소요량 분포", fontsize=11)
        ax2b.legend()
        ax2b.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig2)

    # 그래프 3: 등급 기준 비교
    with tab3:
        fig3, ax3 = plt.subplots(figsize=(10, 5))

        등급목록  = ['+', '1', '2', '3', '4', '5']
        자립률기준 = [120, 100, 80, 60, 40, 20]
        색상      = ['#c39bd3','#ec7063','#f0b27a',
                     '#f9e79f','#a9dfbf','#aed6f1']

        bars = ax3.barh(등급목록, 자립률기준,
                        color=색상, alpha=0.8, edgecolor='gray')
        ax3.axvline(x=pred_자립률, color='black', linewidth=2.5,
                    linestyle='--',
                    label=f'현재 예측값 ({pred_자립률:.1f}%)')

        for bar, val in zip(bars, 자립률기준):
            ax3.text(val+1, bar.get_y()+bar.get_height()/2,
                    f'{val}% 이상', va='center', fontsize=9)

        ax3.set_xlabel("에너지자립률 (%)", fontsize=11)
        ax3.set_title("등급별 자립률 기준 vs 현재 예측값", fontsize=13)
        ax3.legend(fontsize=11)
        ax3.grid(alpha=0.3, axis='x')
        st.pyplot(fig3)

    st.divider()
    st.caption("⚠️ 설계 전 사전 예측이에요! 정확한 등급은 에너지 시뮬레이션 후 확인하세요!")
