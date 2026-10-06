
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# НАСТРОЙКИ
# ============================================================

st.set_page_config(
    page_title="Цифровой паспорт F-01",
    page_icon="⚙️",
    layout="wide"
)

# ============================================================
# ЗАГРУЗКА ML-МОДЕЛЕЙ
# ============================================================

failure_model = joblib.load("failure_model_final.pkl")
rul_model = joblib.load("rul_model_final.pkl")
features = joblib.load("features_final.pkl")

# ============================================================
# СТИЛЬ
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #F4F0E7;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #172A3A;
}

.header {
    padding: 24px 28px;
    border-radius: 18px;
    background: #172A3A;
    color: white;
    margin-bottom: 20px;
}

.header-title {
    font-size: 32px;
    font-weight: 700;
}

.header-sub {
    opacity: 0.75;
    margin-top: 5px;
}

.card {
    background: #FBF8F2;
    border: 1px solid #DED7CB;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 12px;
}

.result {
    background: #FBF8F2;
    border-left: 6px solid #8C6D4E;
    padding: 22px;
    border-radius: 14px;
    margin-top: 12px;
}

.big-number {
    color: #172A3A;
    font-size: 38px;
    font-weight: 750;
}

.small-label {
    color: #716A61;
    font-size: 14px;
}

.normal {
    color: #28784A;
    font-size: 27px;
    font-weight: 700;
}

.attention {
    color: #B77A19;
    font-size: 27px;
    font-weight: 700;
}

.critical {
    color: #A23A34;
    font-size: 27px;
    font-weight: 700;
}

.note {
    font-size: 12px;
    color: #777;
}

div.stButton > button {
    background: #172A3A;
    color: white;
    border-radius: 10px;
    height: 48px;
    font-weight: 700;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# ШАПКА ПАСПОРТА
# ============================================================

st.markdown("""
<div class="header">
    <div class="header-title">
        Цифровой паспорт фланцевого соединения F-01
    </div>

    <div class="header-sub">
        Мониторинг технического состояния • ML-прогноз • поддержка ТОиР
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ПОСТОЯННЫЕ ДАННЫЕ УЗЛА
# ============================================================

st.markdown("### Паспортные данные")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Соединение", "F-01")
c2.metric("Крепёж", "72 × M20×1,5")
c3.metric("Материал", "A4-70")
c4.metric("Целевой преднатяг", "55 кН")

st.divider()

# ============================================================
# ВВОД ДАННЫХ ОСМОТРА
# ============================================================

st.markdown("## Новый осмотр")

st.caption(
    "Введите текущие эксплуатационные параметры и результаты контроля соединения."
)

left, right = st.columns(2)

with left:

    operating_hours = st.number_input(
        "Наработка, ч",
        min_value=0,
        max_value=50000,
        value=4500,
        step=100
    )

    temperature = st.number_input(
        "Температура, °C",
        min_value=-50.0,
        max_value=500.0,
        value=285.0,
        step=5.0
    )

    vibration = st.number_input(
        "Вибрация RMS, мм/с",
        min_value=0.0,
        max_value=30.0,
        value=5.8,
        step=0.1
    )

    wind = st.number_input(
        "Индекс ветровой нагрузки",
        min_value=0.0,
        max_value=5.0,
        value=1.35,
        step=0.05
    )

    thermal_cycles = st.number_input(
        "Количество термоциклов",
        min_value=0,
        max_value=10000,
        value=210,
        step=10
    )

    maintenance_event = st.selectbox(
        "Недавнее техническое обслуживание",
        ["Нет", "Да"]
    )

with right:

    preload_mean = st.number_input(
        "Средний преднатяг, кН",
        min_value=0.0,
        max_value=120.0,
        value=48.5,
        step=0.5
    )

    preload_std = st.number_input(
        "Стандартное отклонение преднатяга, кН",
        min_value=0.0,
        max_value=50.0,
        value=5.2,
        step=0.1
    )

    loose_bolts = st.number_input(
        "Ослабленные болты",
        min_value=0,
        max_value=72,
        value=4,
        step=1
    )

    missing_bolts = st.number_input(
        "Отсутствующие болты",
        min_value=0,
        max_value=72,
        value=0,
        step=1
    )

    max_bolt_force = st.number_input(
        "Максимальное усилие в болте, кН",
        min_value=0.0,
        max_value=150.0,
        value=59.2,
        step=0.5
    )

    inspection_score = st.number_input(
        "Индекс технического состояния",
        min_value=0.0,
        max_value=1.0,
        value=0.72,
        step=0.01
    )

# ============================================================
# РАСЧЁТ ПРОИЗВОДНОГО ПАРАМЕТРА
# ============================================================

if preload_mean > 0:
    preload_cv = preload_std / preload_mean
else:
    preload_cv = 0.0

# ============================================================
# КНОПКА ML-ПРОГНОЗА
# ============================================================

st.markdown("")
run_prediction = st.button("ОЦЕНИТЬ ТЕХНИЧЕСКОЕ СОСТОЯНИЕ")

if run_prediction:

    input_data = pd.DataFrame([{
        "operating_hours": operating_hours,
        "temperature_C": temperature,
        "vibration_rms_mm_s": vibration,
        "wind_load_index": wind,
        "thermal_cycles": thermal_cycles,

        "preload_mean_kN": preload_mean,
        "preload_std_kN": preload_std,
        "preload_cv": preload_cv,

        "loose_bolts": loose_bolts,
        "missing_bolts": missing_bolts,

        "max_bolt_force_kN": max_bolt_force,

        "maintenance_event":
            1 if maintenance_event == "Да" else 0,

        "inspection_score": inspection_score
    }])

    # Порядок признаков строго как при обучении
    input_data = input_data[features]

    # ========================================================
    # ML
    # ========================================================

    probability = failure_model.predict_proba(input_data)[0, 1]

    rul = float(
        rul_model.predict(input_data)[0]
    )

    rul = max(0, rul)

    # ========================================================
    # СТАТУС
    # ========================================================

    if probability < 0.25:

        status = "НОРМА"
        css_status = "normal"

        recommendation = (
            "Продолжить эксплуатацию. "
            "Контроль соединения — по установленному регламенту."
        )

    elif probability < 0.60:

        status = "ВНИМАНИЕ"
        css_status = "attention"

        recommendation = (
            "Рекомендуется дополнительный контроль крепежа, "
            "преднатяга и уровня вибрации."
        )

    else:

        status = "КРИТИЧНО"
        css_status = "critical"

        recommendation = (
            "Требуется внеплановая диагностика соединения "
            "и принятие решения по ТОиР."
        )

    # ========================================================
    # РЕЗУЛЬТАТ
    # ========================================================

    st.divider()
    st.markdown("## Результат анализа")

    r1, r2, r3 = st.columns(3)

with r1:
    st.metric(
        "Вероятность отказа в следующие 500 ч",
        f"{probability * 100:.1f} %"
    )

with r2:
    st.metric(
        "Оценка остаточного ресурса",
        f"{rul:,.0f} ч"
    )

with r3:
    st.metric(
        "Текущее состояние",
        status
    )
    # ========================================================
    # ФАКТОРЫ РИСКА
    # ========================================================

    st.markdown("### Диагностические факторы")

    factors = []

    if vibration > 4.5:
        factors.append("↑ повышенный уровень вибрации")

    if preload_mean < 0.90 * 55:
        factors.append("↓ средний преднатяг относительно целевого")

    if preload_cv > 0.08:
        factors.append("↑ неравномерность преднатяга")

    if loose_bolts > 0:
        factors.append(
            f"⚠ обнаружено ослабленных болтов: {loose_bolts}"
        )

    if missing_bolts > 0:
        factors.append(
            f"⚠ отсутствующих болтов: {missing_bolts}"
        )

    if temperature > 280:
        factors.append("↑ высокая рабочая температура")

    if factors:

        for factor in factors:
            st.write("•", factor)

    else:
        st.write(
            "Выраженных диагностических факторов "
            "по заданным правилам не выявлено."
        )

    # ========================================================
    # РЕКОМЕНДАЦИЯ
    # ========================================================

    st.markdown("### Рекомендация ТОиР")

    st.info(recommendation)

    # ========================================================
    # ДАННЫЕ ДЛЯ ИСТОРИИ
    # ========================================================

    st.markdown("### Запись цифрового паспорта")

    passport_record = pd.DataFrame([{
        "Наработка, ч": operating_hours,
        "Температура, °C": temperature,
        "Вибрация, мм/с": vibration,
        "Средний преднатяг, кН": preload_mean,
        "CV преднатяга, %": preload_cv * 100,
        "Ослаблено": loose_bolts,
        "Отсутствует": missing_bolts,
        "P отказа ≤500 ч, %": probability * 100,
        "RUL, ч": rul,
        "Статус": status
    }])

    st.dataframe(
        passport_record,
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.markdown("""
<div class="note">
ML-прототип разработан на синтетическом наборе эксплуатационных
сценариев. Результаты предназначены для демонстрации подхода.
Перед промышленным применением требуется валидация моделей
на фактических эксплуатационных данных.
</div>
""", unsafe_allow_html=True)
