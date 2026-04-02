import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ─────────────────────────────────────────────
# 0. 한국어 폰트 설정
# ─────────────────────────────────────────────
font_files = [f for f in fm.findSystemFonts() if 'NanumGothic' in f or 'Nanum' in f]
if font_files:
    fm.fontManager.addfont(font_files[0])
    prop = fm.FontProperties(fname=font_files[0])
    matplotlib.rc('font', family=prop.get_name())
else:
    matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 1. 모델 및 데이터 로드
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    m_direct_자립 = joblib.load(os.path.join(BASE_DIR, 'model_direct_자립률.pkl'))
    m_direct_1차  = joblib.load(os.path.join(BASE_DIR, 'model_direct_1차에너지.pkl'))
    m_resid_자립  = joblib.load(os.path.join(BASE_DIR, 'model_resid_자립률.pkl'))
    m_resid_1차   = joblib.load(os.path.join(BASE_DIR, 'model_resid_1차에너지.pkl'))
    le_dict       = joblib.load(os.path.join(BASE_DIR, 'le_dict.pkl'))
    feat_cols     = joblib.load(os.path.join(BASE_DIR, 'feature_columns.pkl'))
    f_자립률      = joblib.load(os.path.join(BASE_DIR, 'interp_자립률.pkl'))
    f_1차에너지   = joblib.load(os.path.join(BASE_DIR, 'interp_1차에너지.pkl'))
    return m_direct_자립, m_direct_1차, m_resid_자립, m_resid_1차, le_dict, feat_cols, f_자립률, f_1차에너지

@st.cache_data
def load_data():
    scatter = pd.read_csv(os.path.join(BASE_DIR, 'df_scatter.csv'))
    return scatter

m_direct_자립, m_direct_1차, m_resid_자립, m_resid_1차, le_dict, feat_cols, f_자립률, f_1차에너지 = load_models()
df_scatter = load_data()

# ─────────────────────────────────────────────
# 2. 예측 함수 (앙상블)
# ─────────────────────────────────────────────
def predict(지역, 용도, 용도구분, 연면적, 창면적비,
            난방, 냉방, 태양광용량, 후면, 밀착, 지열, 열병합):

    태양광비율 = 후면 / 연면적 if 연면적 > 0 else 0

    def enc(col, val):
        le = le_dict[col]
        val_str = str(val)
        if val_str in le.classes_:
            return le.transform([val_str])[0]
        return 0

    row = pd.DataFrame([{
        '신청지역':    enc('신청지역', 지역),
        '건물용도':    enc('건물용도', 용도),
        '건물용도구분': enc('건물용도구분', 용도구분),
        '연면적':      연면적,
        '창면적비':    창면적비,
        '난방방식':    enc('난방방식', 난방),
        '냉방방식':    enc('냉방방식', 냉방),
        '태양광용량':  태양광용량,
        '태양광_후면': 후면,
        '태양광_밀착': 밀착,
        '지열여부':    지열,
        '열병합여부':  열병합,
        '태양광비율':  태양광비율,
    }])[feat_cols]

    # 방법 A: 직접 예측
    direct_자립 = float(m_direct_자립.predict(row)[0])
    direct_1차  = float(m_direct_1차.predict(row)[0])

    # 방법 B: 보간 + 잔차
    보간_자립 = float(np.clip(f_자립률(태양광비율), 0, 200))
    보간_1차  = float(np.clip(f_1차에너지(태양광비율), -250, 400))
    resid_자립 = 보간_자립 + float(m_resid_자립.predict(row)[0])
    resid_1차  = 보간_1차  + float(m_resid_1차.predict(row)[0])

    # 앙상블: 두 방법의 평균
    자립률예측   = float(np.clip((direct_자립 + resid_자립) / 2, 0, 200))
    에너지예측1차 = float(np.clip((direct_1차 + resid_1차) / 2, -250, 400))

    # 규칙 기반 등급 계산
    def calc_grade(자립률, 에너지1차, 용도구분):
        order = ['+', '1', '2', '3', '4', '5']

        if   자립률 >= 120: g1 = '+'
        elif 자립률 >= 100: g1 = '1'
        elif 자립률 >= 80:  g1 = '2'
        elif 자립률 >= 60:  g1 = '3'
        elif 자립률 >= 40:  g1 = '4'
        elif 자립률 >= 20:  g1 = '5'
        else:               g1 = None

        if 용도구분 == '주거용':
            thresholds = [(60,'+'), (90,'1'), (120,'2'),
                          (150,'3'), (190,'4'), (230,'5')]
        else:
            thresholds = [(80,'+'), (140,'1'), (200,'2'),
                          (260,'3'), (320,'4'), (380,'5')]

        g2 = None
        for limit, grade in thresholds:
            if 에너지1차 <= limit:
                g2 = grade
                break

        if g1 is None and g2 is None:
            return '인증불가'
        if g1 is None: return g2
        if g2 is None: return g1
        return g1 if order.index(g1) <= order.index(g2) else g2

    등급예측 = calc_grade(자립률예측, 에너지예측1차, 용도구분)

    return 자립률예측, 에너지예측1차, 태양광비율, 등급예측

# ─────────────────────────────────────────────
# 3. UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="제로에너지 건축물 사전 예측",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 제로에너지 건축물 인증 사전 예측")
st.caption("설계 전 대략적인 에너지자립률과 인증 등급을 예측해드려요!")
st.divider()

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
        ['히트펌프', '보일러', '지역난방', '기타'])

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
        보정면적 = 태양광면적_입력 / 12 * 태양광효율
        st.caption(f"💡 보정 면적: {보정면적:.1f} m²")
    else:
        보정면적 = 태양광면적_입력

    if 태양광타입 == '후면통풍형':
        후면 = 보정면적
        밀착 = 보정면적 * (0.12 / 0.112)
    else:
        밀착 = 보정면적
        후면 = 보정면적 * (0.112 / 0.12)

st.divider()

# ─────────────────────────────────────────────
# 4. 예측 실행
# ─────────────────────────────────────────────
if st.button("🔍 예측하기", type="primary", use_container_width=True):

    용도구분 = "주거용" if '주거' in 건물용도 and '이외' not in 건물용도 else "주거용 이외"

    자립률, 에너지1차, 태양광비율, 등급 = predict(
        지역, 건물용도, 용도구분, 연면적, 창면적비,
        난방방식, 냉방방식, 0, 후면, 밀착,
        1 if 지열여부 == '있음' else 0,
        1 if 열병합여부 == '있음' else 0
    )

    st.subheader("📊 예측 결과")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(label="⚡ 에너지자립률", value=f"{자립률:.1f}%")

    with r2:
        st.metric(label="🔋 1차에너지소요량",
                  value=f"{에너지1차:.1f} kWh/㎡·년",
                  help="⚠️ 참고용 예측값이에요, 오차가 클 수 있어요!")

    with r3:
        등급_이모지 = {'+':'🥇','1':'🥈','2':'🥉',
                      '3':'🏅','4':'🎖️','5':'📋','인증불가':'❌'}
        if 등급 != '인증불가':
            st.metric(label="🏢 예측 등급",
                      value=f"{등급_이모지.get(등급,'')} {등급}등급")
        else:
            st.metric(label="🏢 예측 등급", value="❌ 인증불가")

    if 등급 == '인증불가':
        st.error("⚠️ 현재 조건으로는 인증이 어려워요! 태양광을 늘려보세요!")
    elif 등급 in ['+', '1', '2']:
        st.success("🎉 우수한 등급이 예상돼요!")
    elif 등급 in ['3', '4']:
        st.info("👍 양호한 등급이 예상돼요!")
    else:
        st.warning("💡 태양광을 조금 더 늘리면 더 높은 등급을 받을 수 있어요!")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📈 태양광 면적별 예측", "🗺️ 전체 데이터 분포", "📊 등급 기준 비교"])

    with tab1:
        st.subheader("태양광 면적에 따른 자립률 · 1차에너지 변화")
        면적_범위 = np.linspace(0, min(연면적 * 0.6, 10000), 40)
        자립_목록, 에너지_목록 = [], []
        for 면 in 면적_범위:
            if 태양광타입 == '후면통풍형':
                h, m_val = 면, 면 * (0.12 / 0.112)
            else:
                m_val, h = 면, 면 * (0.112 / 0.12)
            z, e, _, _ = predict(지역, 건물용도, 용도구분, 연면적, 창면적비,
                                  난방방식, 냉방방식, 0, h, m_val,
                                  1 if 지열여부 == '있음' else 0,
                                  1 if 열병합여부 == '있음' else 0)
            자립_목록.append(z)
            에너지_목록.append(e)

        fig1, ax1 = plt.subplots(figsize=(9, 4))
        ax2 = ax1.twinx()
        ax1.plot(면적_범위, 자립_목록, color='steelblue', lw=2, label='자립률 (%)')
        ax2.plot(면적_범위, 에너지_목록, color='tomato', lw=2,
                 linestyle='--', label='1차에너지')
        if 태양광면적_입력 > 0:
            ax1.axvline(보정면적, color='green', linestyle=':',
                        lw=1.5, label=f'현재 {보정면적:.0f}㎡')
        ax1.set_xlabel("태양광 면적 (㎡)")
        ax1.set_ylabel("자립률 (%)", color='steelblue')
        ax2.set_ylabel("1차에너지소요량 (kWh/㎡·년)", color='tomato')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with tab2:
        st.subheader("전체 데이터 분포에서 내 건물 위치")
        sample = df_scatter.sample(min(3000, len(df_scatter)), random_state=42)

        fig2, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(sample['태양광비율'] * 100, sample['에너지자립률'],
                   alpha=0.3, s=10, color='steelblue', label='전체 데이터')
        ax.scatter(태양광비율 * 100, 자립률,
                   color='red', s=200, zorder=5, marker='*', label='내 건물')

        percentile = (df_scatter['에너지자립률'] <= 자립률).mean() * 100
        ax.annotate(f'상위 {100-percentile:.0f}% 수준\n자립률 {자립률:.1f}%',
                    xy=(태양광비율 * 100, 자립률),
                    xytext=(min(태양광비율 * 100 + 3, 45), min(자립률 + 8, 90)),
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_xlabel("태양광비율 (연면적 대비, %)")
        ax.set_ylabel("에너지 자립률 (%)")
        ax.set_title("태양광비율 vs 에너지 자립률")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with tab3:
        st.subheader("등급별 자립률 기준")
        등급_기준 = {'+등급': 120, '1등급': 100, '2등급': 80,
                    '3등급': 60,  '4등급': 40,  '5등급': 20}
        colors = ['gold','silver','#cd7f32','lightblue','lightgreen','lightyellow']

        fig3, ax = plt.subplots(figsize=(7, 4))
        ax.barh(list(등급_기준.keys()), list(등급_기준.values()), color=colors)
        ax.axvline(자립률, color='red', lw=2, linestyle='--',
                   label=f'내 건물: {자립률:.1f}%')
        ax.set_xlabel("에너지 자립률 (%)")
        ax.set_title("등급 기준 vs 예측 자립률")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig3)

    st.divider()
    st.caption("※ 본 예측기는 실제 인증 결과와 ±5~10% 오차가 있을 수 있습니다. 정확한 인증은 공인 평가기관을 통해 확인하세요.")
