import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import datetime
import unicodedata

# ==========================================
# CONFIGURACIÓN GLOBAL KININ PREMIUM
# ==========================================
st.set_page_config(page_title="EPT Anexo 14 - KININ", layout="wide")
COLOR_KININ_DARK = "#002B5B"
COLOR_KININ_LIGHT = "#00A4CC"

# Paleta de Colores OpenCV (Formato BGR)
C_MAIN_BONE = (0, 255, 255)    # Amarillo Neón (Segmento Principal)
C_MAIN_JOINT = (0, 0, 255)     # Rojo (Articulación)
C_PERIPH_BONE = (255, 200, 0)  # Cian/Celeste (Asociado)
C_TRUNK = (150, 150, 150)      # Gris (Eje)
C_PANEL_BG = (40, 20, 10)      # Fondo Panel Oscuro

# --- CSS HACK DEFINITIVO: ANTI-DIMMING & PREMIUM UI ---
st.markdown("""
<style>
    /* Mantiene opacidad 100% y bloquea oscurecimiento */
    .stApp, .main, .block-container, 
    [data-testid="stVerticalBlock"], 
    [data-testid="stHorizontalBlock"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
        pointer-events: auto !important;
    }
    
    /* Oculta spinners y status bar nativos */
    .stSpinner, .st-emotion-cache-1kyxreq, div[data-testid="stStatusWidget"] {
        display: none !important;
    }
    
    /* Estilo Premium para métricas */
    div[data-testid="metric-container"] {
        background-color: rgba(0, 164, 204, 0.05);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00A4CC;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# FUNCIONES MATEMÁTICAS Y CLÍNICAS
# ==========================================
def eliminar_tildes(texto):
    """Limpia caracteres latinos para visualización en OpenCV"""
    if not isinstance(texto, str):
        texto = str(texto)
    texto_normalizado = unicodedata.normalize('NFD', texto)
    return ''.join(c for c in texto_normalizado if unicodedata.category(c) != 'Mn')

def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo interno (0-180) entre 3 puntos"""
    try:
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return int(angle)
    except:
        return 0

def angulo_clinico(p1, p2, p3, tipo):
    """Convierte geometría bruta en Goniometría Anatómica (0 = Neutral)"""
    ang_geom = calcular_angulo(p1, p2, p3)
    if tipo in ["codo", "muneca", "cervical"]:
        return abs(180 - ang_geom) 
    if tipo == "hombro_flex":
        return ang_geom 
    return ang_geom

def sanitizar_pdf(texto):
    """Previene que la librería FPDF colapse con caracteres latinos"""
    if not isinstance(texto, str): 
        texto = str(texto)
    texto = texto.replace("°", " grados")
    texto = texto.replace("ñ", "n").replace("Ñ", "N")
    texto_limpio = eliminar_tildes(texto)
    return texto_limpio.encode('ascii', 'ignore').decode('ascii')

def clasificar_riesgo_circular384(angulo, limite):
    """Aplica criterio de la Circular 384 SUSESO"""
    if angulo >= limite:
        return "PRESENTE (Riesgo Alto)", "Supera Criterio Biomecánico"
    else:
        return "AUSENTE (Riesgo Bajo)", "Dentro de márgenes tolerables"

def clasificar_ciclo(duracion):
    """Clasificación de ciclos Anexo 14"""
    if duracion < 30:
        return "MICROLABOR (<30s)", "⚠️ Alta Repetitividad"
    else:
        return "MACROLABOR (>30s)", "✅ Verificar posturas mantenidas"

def calcular_impacto_ambiental(distancia_km):
    """Calcula KPIs de Sostenibilidad KININ"""
    co2 = distancia_km * 0.21
    agua = (distancia_km / 10 * 4) + 50 
    return co2, agua

# ==========================================
# INTERFAZ Y SIDEBAR
# ==========================================
if os.path.exists("logo_sidebar.png"): 
    st.sidebar.image("logo_sidebar.png", use_container_width=True)

st.title("🛡️ EPT: Estudio de Puesto de Trabajo (Anexo 14)")
st.caption("Plataforma Oficial KININ | v31.0 Uncompressed Titanium")

with st.sidebar:
    st.header("⚙️ Configuración Clínica")
    
    dict_patologias = {
        "Hombro": "Tendinopatía Manguito Rotador",
        "Codo": "Epicondilitis / Epitrocleitis",
        "Muñeca-Mano": "Tendinitis Extensores/Flexores",
        "Mano-Muñeca": "Síndrome Túnel Carpiano",
        "Mano-Pulgar": "Tendinitis De Quervain",
        "Mano-Dedos": "Dedo en Gatillo",
        "Columna Cervical": "Síndrome Tensión Cervical"
    }
    
    segmento_clave = st.selectbox("1. Segmento Principal", list(dict_patologias.keys()))
    lateralidad_eval = st.radio("2. Lado a Evaluar", ["Derecha", "Izquierda", "Bilateral"], horizontal=True)
    
    st.info("🤖 **IA Activa:** Detección de vista automática y telemetría en panel lateral.", icon="🧠")
    
    st.markdown("---")
    st.caption("🌱 Índice de Sostenibilidad KININ")
    distancia_km = st.number_input("Distancia Evitada (km)", 0, 500, 15)

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["🎥 1. Análisis Biomecánico IA", "⚡ 2. Factores Asociados", "📝 3. Informe Clínico 384"])

# ==========================================
# TAB 1: MOTOR BIOMECÁNICO
# ==========================================
with tab1:
    col_u, col_d = st.columns([1, 2.5])
    
    with col_u:
        st.markdown("### 📤 Cargar Evidencia")
        up_file = st.file_uploader("Video (MP4/MOV)", type=['mp4', 'mov'])
        ciclo_seg = st.number_input("Duración Ciclo (seg)", 1, 600, 30)
        
        tipo_c, obs_c = clasificar_ciclo(ciclo_seg)
        if "MICRO" in tipo_c: 
            st.warning(f"📌 {tipo_c}")
            st.caption(obs_c)
        else: 
            st.success(f"📌 {tipo_c}")
            st.caption(obs_c)

    with col_d:
        st_frame = st.empty() 
        
        if up_file:
            # ID Único de caché
            file_id = f"{up_file.name}_{up_file.size}_{segmento_clave}_{lateralidad_eval}"
            
            if st.session_state.get('last_file_id') == file_id and 'last_frame' in st.session_state:
                st_frame.image(st.session_state.last_frame, channels="BGR", caption="✅ Análisis Biomecánico Completo")
            
            else:
                # Creación y cierre seguro de Tempfile para evitar Crash en Windows
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(up_file.read())
                tfile.close() 
                
                cap = cv2.VideoCapture(tfile.name)
                history_list = []
                my_bar = st.progress(0, text="Calibrando Goniometría Anatómica...")
                
                mp_pose = mp.solutions.pose
                mp_hands = mp.solutions.hands
                
                MODO_MANO = any(x in segmento_clave for x in ["Pulgar", "Dedos"])
                
                # Evitar ZeroDivisionError
                total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
                curr_frame = 0
                vista_detectada = "Calculando..."
                
                # --- LÓGICA MANOS / MICROMOVIMIENTOS ---
                if MODO_MANO:
                    with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.6) as hands:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            
                            curr_frame += 1
                            if curr_frame % 3 != 0: continue
                            
                            my_bar.progress(min(curr_frame / total_frames, 1.0), text="Analizando Micro-movimientos distales...")
                            frame = cv2.resize(frame, (640, 480))
                            
                            # Crear Panel Lateral de Telemetría
                            side_panel = np.full((480, 360, 3), C_PANEL_BG, dtype=np.uint8)
                            cv2.putText(side_panel, "TELEMETRIA KININ", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                            cv2.line(side_panel, (20, 50), (340, 50), (255, 255, 255), 1)
                            
                            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            angulos_frame = {}
                            
                            if res.multi_hand_landmarks:
                                for hand_idx, hand_lm in enumerate(res.multi_hand_landmarks):
                                    lm = hand_lm.landmark
                                    
                                    # Dibujo de esqueleto mano
                                    mp.solutions.drawing_utils.draw_landmarks(
                                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                                        mp.solutions.drawing_utils.DrawingSpec(color=C_MAIN_JOINT, thickness=2, circle_radius=3),
                                        mp.solutions.drawing_utils.DrawingSpec(color=C_MAIN_BONE, thickness=3)
                                    )
                                    
                                    # Auto-Detección de Vista
                                    w_hand = np.linalg.norm(np.array([lm[5].x, lm[5].y]) - np.array([lm[17].x, lm[17].y]))
                                    l_hand = np.linalg.norm(np.array([lm[0].x, lm[0].y]) - np.array([lm[9].x, lm[9].y]))
                                    ratio = w_hand / (l_hand + 1e-6)
                                    vista_detectada = "Superior/Palmar" if ratio > 0.45 else "Lateral/Sagital"
                                    
                                    prefix = f"M{hand_idx+1}-" 
                                    
                                    if "Pulgar" in segmento_clave:
                                        angulos_frame[f"{prefix}Abd Pulgar"] = calcular_angulo([lm[5].x, lm[5].y], [lm[0].x, lm[0].y], [lm[4].x, lm[4].y])

                                history_list.append(angulos_frame)
                                
                                # Renderizado en Panel Lateral
                                cv2.putText(side_panel, f"Vista IA: {vista_detectada}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                                y = 130
                                for k, v in angulos_frame.items():
                                    cv2.putText(side_panel, f"{eliminar_tildes(k)}: {v} gr", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_MAIN_BONE, 2)
                                    y += 40
                            
                            # Unión visual de Video + Panel
                            final_frame = np.hstack((frame, side_panel))
                            st_frame.image(final_frame, channels="BGR")
                            st.session_state.last_frame = final_frame 

                # --- LÓGICA CUERPO / MACROLABORES ---
                else: 
                    with mp_pose.Pose(model_complexity=2, min_detection_confidence=0.7) as pose:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            
                            curr_frame += 1
                            if curr_frame % 5 != 0: continue
                            
                            my_bar.progress(min(curr_frame / total_frames, 1.0), text="Goniometría Anatómica y Cadena Cinética...")
                            frame = cv2.resize(frame, (640, 480))
                            
                            # Crear Panel Lateral de Telemetría
                            side_panel = np.full((480, 360, 3), C_PANEL_BG, dtype=np.uint8)
                            cv2.putText(side_panel, "TELEMETRIA KININ", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                            cv2.line(side_panel, (20, 50), (340, 50), (255, 255, 255), 1)
                            
                            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            
                            if res.pose_landmarks:
                                angulos_frame = {}
                                lm = res.pose_landmarks.landmark
                                
                                # Eje Cervical Unificado
                                mid_sh = [(lm[11].x + lm[12].x)/2, (lm[11].y + lm[12].y)/2]
                                mid_ea = [(lm[7].x + lm[8].x)/2, (lm[7].y + lm[8].y)/2]
                                mid_hi = [(lm[23].x + lm[24].x)/2, (lm[23].y + lm[24].y)/2]
                                
                                flex_cervical = angulo_clinico(mid_hi, mid_sh, mid_ea, "cervical")
                                
                                # Configuración de lateralidad (Incluye índice para vector mano)
                                sides_to_check = []
                                if lateralidad_eval in ["Derecha", "Bilateral"]: 
                                    sides_to_check.append(("D", 12, 14, 16, 24, 20)) 
                                if lateralidad_eval in ["Izquierda", "Bilateral"]: 
                                    sides_to_check.append(("I", 11, 13, 15, 23, 19))
                                
                                y_text_panel = 90
                                
                                # Dibujo de Eje de Gravedad (Tronco)
                                cv2.line(frame, (int(mid_sh[0]*640), int(mid_sh[1]*480)), 
                                         (int(mid_hi[0]*640), int(mid_hi[1]*480)), C_TRUNK, 3, cv2.LINE_AA)

                                # Dibujo Cervical y Asociación
                                if "Cervical" in segmento_clave:
                                    angulos_frame["Flexion Cervical"] = flex_cervical
                                    cv2.line(frame, (int(mid_sh[0]*640), int(mid_sh[1]*480)), 
                                             (int(mid_ea[0]*640), int(mid_ea[1]*480)), C_MAIN_BONE, 5, cv2.LINE_AA)
                                elif "Hombro" in segmento_clave:
                                    angulos_frame["Flexion Cervical (Asoc)"] = flex_cervical
                                    cv2.line(frame, (int(mid_sh[0]*640), int(mid_sh[1]*480)), 
                                             (int(mid_ea[0]*640), int(mid_ea[1]*480)), C_PERIPH_BONE, 2, cv2.LINE_AA)

                                # Análisis por segmento
                                for side_prefix, idx_h, idx_c, idx_m, idx_k, idx_hand in sides_to_check:
                                    h = [lm[idx_h].x, lm[idx_h].y]
                                    c = [lm[idx_c].x, lm[idx_c].y]
                                    m = [lm[idx_m].x, lm[idx_m].y]
                                    k = [lm[idx_k].x, lm[idx_k].y]
                                    hand = [lm[idx_hand].x, lm[idx_hand].y]
                                    
                                    h_px, h_py = int(h[0]*640), int(h[1]*480)
                                    c_px, c_py = int(c[0]*640), int(c[1]*480)
                                    m_px, m_py = int(m[0]*640), int(m[1]*480)
                                    hand_px, hand_py = int(hand[0]*640), int(hand[1]*480)
                                    
                                    if "Hombro" in segmento_clave:
                                        cv2.line(frame, (h_px, h_py), (c_px, c_py), C_MAIN_BONE, 5, cv2.LINE_AA)
                                        cv2.circle(frame, (h_px, h_py), 8, C_MAIN_JOINT, -1)
                                        ang_flex = angulo_clinico([h[0], h[1]+0.5], h, c, "hombro_flex")
                                        angulos_frame[f"{side_prefix}-Flex Hombro"] = ang_flex
                                        
                                    elif "Codo" in segmento_clave:
                                        cv2.line(frame, (h_px, h_py), (c_px, c_py), C_MAIN_BONE, 4, cv2.LINE_AA)
                                        cv2.line(frame, (c_px, c_py), (m_px, m_py), C_MAIN_BONE, 4, cv2.LINE_AA)
                                        cv2.circle(frame, (c_px, c_py), 8, C_MAIN_JOINT, -1)
                                        angulos_frame[f"{side_prefix}-Flex Codo"] = angulo_clinico(h, c, m, "codo")

                                    elif "Muñeca" in segmento_clave:
                                        cv2.line(frame, (c_px, c_py), (m_px, m_py), C_MAIN_BONE, 4, cv2.LINE_AA)
                                        cv2.line(frame, (m_px, m_py), (hand_px, hand_py), C_MAIN_BONE, 4, cv2.LINE_AA)
                                        cv2.circle(frame, (m_px, m_py), 8, C_MAIN_JOINT, -1)
                                        
                                        ang_muneca = angulo_clinico(c, m, hand, "muneca")
                                        # Lógica clínica: Si la mano está más alta que la muñeca es extensión.
                                        label = "Ext Muneca" if hand[1] < m[1] else "Flex Muneca"
                                        angulos_frame[f"{side_prefix}-{label}"] = ang_muneca

                                # Escribir datos en el Panel Lateral
                                for k_txt, v in angulos_frame.items():
                                    if "(Asoc)" in k_txt:
                                        color_txt, size_txt = C_PERIPH_BONE, 0.6
                                    else:
                                        color_txt, size_txt = C_MAIN_BONE, 0.75
                                        
                                    texto_limpio = eliminar_tildes(k_txt)
                                    cv2.putText(side_panel, f"{texto_limpio}: {v} gr", (20, y_text_panel), cv2.FONT_HERSHEY_SIMPLEX, size_txt, color_txt, 2)
                                    y_text_panel += 40

                                history_list.append(angulos_frame)
                            
                            # Unión Visual
                            final_frame = np.hstack((frame, side_panel))
                            st_frame.image(final_frame, channels="BGR")
                            st.session_state.last_frame = final_frame 
                
                my_bar.empty()
                st.session_state.angle_data = pd.DataFrame(history_list)
                st.session_state.last_file_id = file_id 
                st.rerun()

# ==========================================
# TAB 2: VIBRACIONES Y BORG
# ==========================================
with tab2:
    st.subheader("〰️ Exposición a Vibraciones")
    aplica_vibra = st.checkbox("¿El puesto presenta exposición a vibraciones?", value=False)
    
    if aplica_vibra:
        st.markdown("---")
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            tipo_vibra = st.radio("Tipo de Exposición", ["Mano-Brazo", "Cuerpo Entero"])
            modo_entrada = st.radio("Origen de Datos", ["Base INSST Referencial", "Medición en Terreno"])
            
            if modo_entrada == "Base INSST Referencial":
                if "Mano" in tipo_vibra:
                    db = {"Taladro Percutor": 12.0, "Martillo Neumático": 18.5, "Esmeril Angular": 6.5, "Lijadora": 4.5, "Llave Impacto": 13.0, "Motosierra": 5.5}
                else:
                    db = {"Camión Minero (CAEX)": 0.85, "Excavadora": 1.15, "Retroexcavadora": 0.95, "Bulldozer": 1.25, "Grúa Horquilla": 1.00, "Bus Transporte": 0.70}
                item = st.selectbox("Equipo Analizado", list(db.keys()))
                a_eq = db[item]
            else:
                a_eq = st.number_input("Aceleración Equivalente (m/s²)", 0.0, 50.0, 2.5)
                item = "Valor Manual"
            
            t_exp = st.slider("Tiempo Exposición Diario (Horas)", 0.1, 12.0, 4.0)
            
        with c2:
            a8 = a_eq * ((t_exp/8)**0.5)
            limite = 5.0 if "Mano" in tipo_vibra else 1.15
            accion = 2.5 if "Mano" in tipo_vibra else 0.5
            
            if a8 >= limite:
                estado, color = "CRÍTICO", "#dc3545"
            elif a8 >= accion:
                estado, color = "ALERTA", "#ffc107"
            else:
                estado, color = "BAJO", "#28a745"
                
            st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 10px solid {color};">
                <h2 style="color: white; margin:0;">A(8) = {a8:.2f} m/s²</h2>
                <p style="color: {color}; margin:0; font-weight:bold;">{estado}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.vib_final = f"{tipo_vibra} | A(8): {a8:.2f} | {estado}"
    else:
        st.session_state.vib_final = "No Aplica"

    st.markdown("---")
    st.subheader("🥵 Escala BORG CR-10")
    b_val = st.select_slider("Nivel de Esfuerzo", options=list(range(11)), value=0)
    
    borg_map = {
        0: ("🛌", "#2980b9"), 1: ("😃", "#2ecc71"), 2: ("😌", "#2ecc71"), 
        3: ("🙂", "#f1c40f"), 4: ("😐", "#f1c40f"), 5: ("😟", "#e67e22"), 
        6: ("😣", "#e67e22"), 7: ("😫", "#d35400"), 8: ("🥵", "#d35400"), 
        9: ("😰", "#c0392b"), 10: ("💀", "#8b0000")
    }
    emoji, bg = borg_map[b_val]
    
    st.markdown(f"""
    <div style="background-color: {bg}; padding: 15px; border-radius: 12px; display: flex; align-items: center; justify-content: center; gap: 25px; max-width: 600px; margin: auto;">
        <div style="font-size: 50px; line-height: 1;">{emoji}</div>
        <div style="text-align: left;">
            <div style="font-size: 32px; font-weight: bold; color: white; line-height: 1;">Nivel {b_val}</div>
            <div style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 5px;">Esfuerzo Percibido</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.borg_final = f"{b_val}/10"

# ==========================================
# TAB 3: INFORME CLÍNICO Y PDF
# ==========================================
with tab3:
    st.markdown("### 📝 Datos Administrativos")
    
    with st.expander("1. Identificación Empleador y Trabajador", expanded=True):
        c1, c2 = st.columns(2)
        with c1: 
            empresa = st.text_input("Empresa", "MUELLAJE S.A.")
            rut_e = st.text_input("RUT Emp.", "90.123.456-7")
        with c2: 
            trabajador = st.text_input("Trabajador", "Jorge Bastias")
            rut_t = st.text_input("RUT Trab.", "9.424.449-8")
        
        c_ex1, c_ex2 = st.columns(2)
        with c_ex1: 
            puesto = st.text_input("Cargo", "Movilizador")
            antiguedad = st.text_input("Antigüedad", "3 años")
        with c_ex2: 
            lat_trabajador = st.radio("Dominancia", ["Diestro", "Zurdo"], horizontal=True)

    with st.expander("2. Organización del Trabajo"):
        desc_tareas = st.text_area("Tareas", "Levantamiento de carga manual...", height=70)
    
    with st.expander("3. Conclusiones Ergonómicas"):
        conclusiones = st.text_area("Dictamen", "Se observa riesgo biomecánico sostenido...", height=70)

    st.markdown("---")
    ce, cf = st.columns(2)
    with ce: 
        evaluador = st.text_input("Evaluador", "Gonzalo Ortega")
    with cf: 
        fecha_eval = st.date_input("Fecha", datetime.date.today())

    if 'angle_data' in st.session_state:
        st.markdown("### 📊 Dashboards Clínicos")
        df = st.session_state.angle_data
        
        # Filtro de columnas principales vs asociadas
        m_cols = [c for c in df.columns if "(Asoc)" not in c]
        a_cols = [c for c in df.columns if "(Asoc)" in c]
        
        # Gráfico Principal
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.axhspan(0, 60, color='green', alpha=0.1)
        for col in m_cols: 
            ax1.plot(df[col], linewidth=2.5, label=col)
        ax1.axhline(y=60, color='red', linestyle='--')
        ax1.set_title(f"Segmento Principal: {dict_patologias[segmento_clave]}", fontsize=11)
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.savefig("t_main.png", bbox_inches='tight', dpi=100)
        plt.close(fig1)

        # Gráfico Asociado
        if a_cols:
            fig2, ax2 = plt.subplots(figsize=(10, 2.5))
            ax2.axhspan(0, 45, color='blue', alpha=0.1)
            for col in a_cols: 
                ax2.plot(df[col], linewidth=2, linestyle=':', color='orange', label=col)
            ax2.axhline(y=45, color='red', linestyle='--')
            ax2.set_title("Factores Asociados (Anexo 14)", fontsize=11)
            ax2.legend(loc='upper right', fontsize='small')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.savefig("t_asoc.png", bbox_inches='tight', dpi=100)
            plt.close(fig2)

        # Cálculo de Criterio
        max_ang = df[m_cols[0]].max() if m_cols and not df.empty else 0
        r_msg, d_msg = clasificar_riesgo_circular384(max_ang, 60 if "Hombro" in segmento_clave else 45)
        
        ck1, ck2 = st.columns(2)
        ck1.metric("Amplitud Máxima", f"{int(max_ang)}°")
        ck2.metric("Calificación (Circ. 384)", r_msg, delta=d_msg, delta_color="inverse")
        
        co2, agua = calcular_impacto_ambiental(distancia_km)
        
        # Generación de PDF
        if st.button("📄 GENERAR INFORME OFICIAL PDF", type="primary"):
            pdf = FPDF()
            pdf.add_page()
            
            # Header KININ
            pdf.set_fill_color(0, 43, 91)
            pdf.rect(0, 0, 210, 25, 'F')
            if os.path.exists("logo_pdf.png"): 
                pdf.image("logo_pdf.png", x=10, y=5, w=30)
            
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(255, 255, 255)
            pdf.set_xy(50, 8)
            pdf.cell(0, 10, sanitizar_pdf(f"ESTUDIO EPT: {segmento_clave.upper()}"), 0, 1, 'L')
            
            pdf.ln(15)
            pdf.set_text_color(0,0,0)
            
            # Helper para tablas
            def c_pair(l, v, w1=45, w2=50): 
                pdf.set_font("Arial", 'B', 9)
                pdf.set_fill_color(230, 230, 230)
                pdf.cell(w1, 7, sanitizar_pdf(l), 1, 0, 'L', 1)
                pdf.set_font("Arial", '', 9)
                pdf.cell(w2, 7, sanitizar_pdf(v), 1, 0, 'L', 0)

            # 1. Identificación
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, "1. IDENTIFICACION GENERAL", 0, 1)
            c_pair("Empresa", empresa); c_pair("RUT", rut_e); pdf.ln()
            c_pair("Trabajador", trabajador); c_pair("RUT", rut_t); pdf.ln()
            c_pair("Cargo", puesto); c_pair("Antiguedad", antiguedad); pdf.ln()
            
            # 2. Análisis
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, "2. ESTUDIO BIOMECANICO", 0, 1)
            if os.path.exists("t_main.png"): 
                pdf.image("t_main.png", x=10, w=190)
            if a_cols and os.path.exists("t_asoc.png"): 
                pdf.ln(2)
                pdf.image("t_asoc.png", x=10, w=190)
            
            # 3. Circular 384
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, "3. PRE-CALIFICACION (CIRCULAR 384)", 0, 1)
            if "Alto" in r_msg:
                pdf.set_fill_color(255, 240, 240)
            else:
                pdf.set_fill_color(240, 255, 240)
            pdf.multi_cell(0, 7, sanitizar_pdf(f"Criterio de Exposicion: {r_msg}\nAmplitud Maxima: {int(max_ang)} grados"), 1, 'L', 1)
            
            # 4. Conclusiones
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, "4. CONCLUSIONES", 0, 1)
            pdf.set_font("Arial", '', 9)
            pdf.multi_cell(0, 6, sanitizar_pdf(conclusiones), 1)
            
            # 5. Firma Digital
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 5, "___________________________________________________", 0, 1, 'C')
            pdf.cell(0, 6, sanitizar_pdf(f"Firma Digital - {evaluador}"), 0, 1, 'C')
            pdf.set_font("Arial", '', 8)
            pdf.cell(0, 5, sanitizar_pdf(f"Ergonomo Responsable | Fecha: {fecha_eval}"), 0, 1, 'C')
            
            # Footer Sostenibilidad
            pdf.set_y(-25)
            pdf.set_font("Arial", 'I', 8)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, sanitizar_pdf(f"Indice de Sostenibilidad KININ - CO2 Evitado: {co2:.2f} kg | Agua Ahorrada: {agua:.1f} L"), 0, 1, 'C')
            
            # Guardado
            pdf_name = f"EPT_{trabajador}.pdf"
            pdf.output(pdf_name)
            with open(pdf_name, "rb") as f: 
                st.download_button("⬇️ DESCARGAR INFORME OFICIAL", f, file_name=pdf_name, mime="application/pdf")