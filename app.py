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
    m_자립      = joblib.load(os.path.join(BASE_DIR, 'model_자립률.pkl'))
    m_1차       = joblib.load(os.path.join(BASE_DIR, 'model_1차에너지.pkl'))
    m_등급      = joblib.load(os.path.join(BASE_DIR, 'model_등급.pkl'))
    le_등급     = joblib.load(os.path.join(BASE_DIR, 'le_등급.pkl'))
    le_dict     = joblib.load(os.path.join(BASE_DIR, 'le_dict.pkl'))
    feat_cols   = joblib.load(os.path.join(BASE_DIR, 'feature_columns.pkl'))
    f_자립률    = joblib.load(os.path.join(BASE_DIR, 'interp_자립률.pkl'))
    f_1차에너지  = joblib.load(os.path.join(BASE_DIR, 'interp_1차에너지.pkl'))
    return m_자립, m_1차, m_등급, le_등급, le_dict, feat_cols, f_자립률, f_1차에너지

@st.cache_data
def load_data():
    scatter = pd.read_csv(os.path.join(BASE_DIR, 'df_scatter.csv'))
    full    = pd.read_csv(os.path.join(BASE_DIR, 'df_model_cleaned.csv'))
    return scatter, full

m_자립, m_1차, m_등급, le_등급, le_dict, feat_cols, f_자립률, f_1차에너지 = load_models()
df_scatter, df_full = load_data()

# ─────────────────────────────────────────────
# 2. 예측 함수
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

    # 보간 + 잔차 합산 방식
    보간_자립  = float(np.clip(f_자립률(태양광비율),    0,   150))
    보간_1차   = float(np.clip(f_1차에너지(태양광비율), -50, 300))

    잔차_자립  = float(m_자립.predict(row)[0])
    잔차_1차   = float(m_1차.predict(row)[0])

    자립률예측   = float(np.clip(보간_자립  + 잔차_자립, 0,   150))
    에너지예측1차 = float(np.clip(보간_1차   + 잔차_1차,  -50, 300))

    # XGB 등급 분류 모델 사용
    등급enc  = m_등급.predict(row)[0]
    등급예측  = le_등급.inverse_transform([등급enc])[0]

    return 자립률예측, 에너지예측1차, 태양광비율, 등급예측

# ─────────────────────────────────────────────
# 3. UI 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="제로에너지 자립률 예측기", layout="wide")
st.title("🏢 제로에너지 건축물 자립률 예측기")
st.markdown("건물 정보를 입력하면 **에너지 자립률**, **1차에너지소요량**, **인증 등급**을 예측합니다.")

with st.sidebar:
    st.header("📋 건물 정보 입력")

    지역_목록 = list(le_dict['신청지역'].classes_)
    용도_목록 = list(le_dict['건물용도'].classes_)

    지역    = st.selectbox("지역", 지역_목록)
    용도    = st.selectbox("건물용도", 용도_목록)
    용도구분 = st.radio("건물용도구분", ["주거용 이외", "주거용"])
    연면적   = st.number_input("연면적 (㎡)", min_value=100, max_value=500000,
                               value=5000, step=100)
    창면적비 = st.slider("창면적비 (%)", min_value=0, max_value=100, value=25)

    st.markdown("---")
    난방 = st.selectbox("난방방식", ["히트펌프", "보일러", "지역난방", "기타"])
    냉방 = st.selectbox("냉방방식", ["압축식", "흡수식", "냉방없음", "기타"])

    st.markdown("---")
    태양광타입 = st.radio("태양광 패널 유형", ["후면개방형", "밀착형"])
    태양광면적 = st.number_input("태양광 패널 면적 (㎡)", min_value=0,
                                 max_value=50000, value=0, step=10)
    효율      = st.slider("태양광 효율 보정 계수", min_value=0.80,
                          max_value=1.20, value=1.00, step=0.01)
    태양광용량 = st.number_input("태양광 용량 (kW)", min_value=0,
                                 max_value=50000, value=0, step=10)

    후면 = 태양광면적 * 효율 if 태양광타입 == "후면개방형" else 0
    밀착 = 태양광면적 * 효율 if 태양광타입 == "밀착형"    else 0

    st.markdown("---")
    지열  = st.checkbox("지열 설비 있음")
    열병합 = st.checkbox("열병합 설비 있음")

    예측버튼 = st.button("🔍 예측하기", use_container_width=True)

# ─────────────────────────────────────────────
# 4. 예측 결과
# ─────────────────────────────────────────────
if 예측버튼:
    자립률, 에너지1차, 태양광비율, 등급 = predict(
        지역, 용도, 용도구분, 연면적, 창면적비,
        난방, 냉방, 태양광용량, 후면, 밀착,
        int(지열), int(열병합)
    )

    등급_이모지 = {'+':'🏆','1':'🥇','2':'🥈','3':'🥉',
                  '4':'🌿','5':'🌱','인증불가':'❌'}.get(등급, '❓')

    # ── 결과 메트릭 ───────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("⚡ 에너지 자립률",    f"{자립률:.1f}%")
    col2.metric("🔋 1차에너지소요량",  f"{에너지1차:.1f} kWh/㎡·년")
    col3.metric("🏅 예측 등급",        f"{등급_이모지} {등급}")

    # ── 자립률 기반 안내 메시지 ───────────────
    if 자립률 >= 100:
        st.success("✅ 에너지 자립률 100% 이상! 제로에너지 건축 1등급 가능성 있습니다.")
    elif 자립률 >= 60:
        st.info(f"💡 자립률 {자립률:.1f}% - 제로에너지 건축 3등급 수준입니다.")
    elif 자립률 >= 20:
        st.warning(f"⚠️ 자립률 {자립률:.1f}% - 태양광 설비 확대를 검토해보세요.")
    else:
        st.error(f"❌ 자립률 {자립률:.1f}% - 재생에너지 설비 도입이 필요합니다.")

    # ── 탭 구성 ──────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 태양광 면적별 예측", "🗺️ 전체 데이터 분포", "📊 등급 기준 비교"])

    # ── Tab 1: 태양광 면적 변화 ───────────────
    with tab1:
        st.subheader("태양광 면적에 따른 자립률 · 1차에너지 변화")
        면적_범위 = np.linspace(0, min(연면적 * 0.6, 10000), 40)
        자립_목록, 에너지_목록 = [], []
        for 면 in 면적_범위:
            h = 면 * 효율 if 태양광타입 == "후면개방형" else 0
            m = 면 * 효율 if 태양광타입 == "밀착형"    else 0
            z, e, _, _ = predict(지역, 용도, 용도구분, 연면적, 창면적비,
                                  난방, 냉방, 태양광용량, h, m,
                                  int(지열), int(열병합))
            자립_목록.append(z)
            에너지_목록.append(e)

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
        ax1.plot(면적_범위, 자립_목록,  color='steelblue', lw=2, label='자립률 (%)')
        ax2.plot(면적_범위, 에너지_목록, color='tomato', lw=2,
                 linestyle='--', label='1차에너지')
        ax1.axvline(태양광면적, color='green', linestyle=':',
                    lw=1.5, label=f'현재 {태양광면적}㎡')
        ax1.set_xlabel("태양광 면적 (㎡)")
        ax1.set_ylabel("자립률 (%)", color='steelblue')
        ax2.set_ylabel("1차에너지소요량", color='tomato')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        st.pyplot(fig1)

    # ── Tab 2: 산점도 ─────────────────────────
    with tab2:
        st.subheader("전체 데이터 분포에서 내 건물 위치")
        sample = df_scatter.sample(min(3000, len(df_scatter)), random_state=42)

        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sample['태양광비율'] * 100, sample['에너지자립률'],
                   alpha=0.3, s=10, color='steelblue', label='전체 데이터')
        ax.scatter(태양광비율 * 100, 자립률,
                   color='red', s=200, zorder=5, marker='*', label='내 건물')

        # 백분위 계산
        percentile = (df_scatter['에너지자립률'] <= 자립률).mean() * 100
        ax.annotate(f'상위 {100-percentile:.0f}% 수준\n자립률 {자립률:.1f}%',
                    xy=(태양광비율 * 100, 자립률),
                    xytext=(태양광비율 * 100 + 2, 자립률 + 5),
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_xlabel("태양광비율 (연면적 대비, %)")
        ax.set_ylabel("에너지 자립률 (%)")
        ax.set_title("태양광비율 vs 에너지 자립률")
        ax.legend()
        st.pyplot(fig2)

    # ── Tab 3: 등급 기준 ──────────────────────
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
        st.pyplot(fig3)

    # ── 하단 안내 ─────────────────────────────
    st.markdown("---")
    st.caption("※ 본 예측기는 실제 인증 결과와 ±5~10% 오차가 있을 수 있습니다. 정확한 인증은 공인 평가기관을 통해 확인하세요.")
