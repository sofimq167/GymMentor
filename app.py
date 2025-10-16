import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
import pickle
import numpy as np
from tensorflow import keras
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from PIL import Image
import cnn_module

load_dotenv()

# Configurar página
st.set_page_config(page_title="GymMentor", page_icon="💪", layout="wide")

# Inicializar cliente
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Cargar modelo ML
with open('nivel_classifier.pkl', 'rb') as f:
    clf_nivel = pickle.load(f)

# Cargar modelo LSTM
lstm_model = keras.models.load_model('adherencia_lstm.keras')
# Título
st.title("GymMentor - Tu Entrenador Personal IA")
st.markdown("---")

# Sidebar con formulario
with st.sidebar:
    st.header("Tu Perfil")
    dias = st.slider("Días disponibles por semana:", 1, 7, 4)
    objetivo = st.selectbox("Objetivo:", 
        ["Ganar masa muscular", "Perder grasa", "Mantener forma", "Fuerza"])
    nivel = st.selectbox("Nivel:", 
        ["Principiante", "Intermedio", "Avanzado"])
    
    # Detector de equipamiento con CNN
    st.markdown("---")
    st.subheader("Detectar Equipos")
    
    uploaded_files = st.file_uploader(
        "Sube foto(s) de tu espacio de entrenamiento",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Puedes subir varias fotos para detectar más equipos"
    )
    
    # Inicializar estado del análisis
    if 'analisis_completo' not in st.session_state:
        st.session_state.analisis_completo = False
    
    if uploaded_files:
        # Botón para analizar
        if st.button("Analizar todas las imágenes", key="analizar_todas"):
            # Inicializar lista de equipos detectados
            todos_equipos = []
            todas_confianzas = {}
            
            with st.spinner("Analizando imágenes"):
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Mostrar imagen
                        image = Image.open(uploaded_file)
                        
                        # Analizar con CNN
                        resultado = cnn_module.analizar_imagen(image)
                        equipos_detectados = resultado['equipos']
                        confianzas = resultado['confianzas']
                        
                        # Agregar a la lista total (sin duplicar)
                        for eq in equipos_detectados:
                            if eq not in todos_equipos:
                                todos_equipos.append(eq)
                                todas_confianzas[eq] = confianzas.get(eq, 0)
                        
                    except Exception as e:
                        st.error(f"Error en foto {idx+1}: {e}")
                
                # Guardar resultados en session_state
                if todos_equipos:
                    equipos_para_agregar = cnn_module.equipos_a_lista(todos_equipos)
                    st.session_state.equipos_sugeridos = equipos_para_agregar
                    st.session_state.confianzas_detectadas = todas_confianzas
                    st.session_state.analisis_completo = True
                    st.rerun()
        
        # Mostrar preview de las fotos
        if not st.session_state.analisis_completo:
            st.write("**Vista previa:**")
            cols = st.columns(min(len(uploaded_files), 3))
            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx % 3]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Foto {idx+1}", use_container_width=True)
    
    # Mostrar resultados si el análisis está completo
    if st.session_state.analisis_completo and 'equipos_sugeridos' in st.session_state:
        st.success("Análisis completado!")
        
        # Mostrar equipos detectados
        st.write("**Equipos detectados:**")
        for eq in st.session_state.equipos_sugeridos:
            conf = st.session_state.confianzas_detectadas.get(eq, 0) * 100
            st.write(f"- {eq}")
        
        st.info(f"{len(st.session_state.equipos_sugeridos)} equipos detectados")
        
        # Botones para agregar o ignorar (AHORA PERSISTEN)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Agregar al selector", key="agregar_detectados_btn"):
                # Inicializar si no existe
                if 'equipos_seleccionados' not in st.session_state:
                    st.session_state.equipos_seleccionados = []
                
                # Agregar equipos detectados
                for eq in st.session_state.equipos_sugeridos:
                    if eq not in st.session_state.equipos_seleccionados:
                        st.session_state.equipos_seleccionados.append(eq)
                
                # Limpiar análisis
                st.session_state.analisis_completo = False
                st.session_state.equipos_sugeridos = []
                
                st.success("¡Equipos agregados!")
                st.rerun()
        
        with col2:
            if st.button("❌ Ignorar", key="ignorar_detectados_btn"):
                st.session_state.analisis_completo = False
                st.session_state.equipos_sugeridos = []
                st.info("Análisis descartado")
                st.rerun()
    
    # Predictor automático de nivel con ML
    st.markdown("---")
    with st.expander("Predicción automática de nivel"):
        st.write("Responde estas preguntas y la IA predecirá tu nivel:")
        
        años = st.number_input("¿Cuántos años llevas entrenando?", 
                               min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        
        conocimiento = st.slider("Conocimiento de técnica (1-10):", 1, 10, 5,
                                help="¿Qué tan bien conoces la técnica de ejercicios básicos?")
        
        dominadas = st.number_input("¿Cuántas dominadas puedes hacer seguidas?", 
                                    min_value=0, max_value=50, value=5,
                                    help="Si no puedes hacer dominadas, pon 0")
        
        sigue_rutina = st.radio("¿Sigues una rutina estructurada?",
                               ["Sí, tengo un plan", "No, improviso cada día"],
                               help="¿Tienes un programa de entrenamiento o vas sin plan?")
        sigue_rutina_val = 1 if sigue_rutina == "Sí, tengo un plan" else 0
        
        if st.button("🔮 Predecir mi nivel", use_container_width=True):
            # Hacer predicción con nuevas features
            X = np.array([[años, dias, conocimiento, dominadas, sigue_rutina_val]])
            pred = clf_nivel.predict(X)[0]
            prob = clf_nivel.predict_proba(X)[0][pred]
            niveles_map = ["Principiante", "Intermedio", "Avanzado"]
            nivel_predicho = niveles_map[pred]
            
            # Mostrar resultado
            st.success(f"**Nivel predicho:** {nivel_predicho}")
            st.info(f"Confianza del modelo: {prob*100:.0f}%")
            
            # Actualizar la variable nivel
            nivel = nivel_predicho
            st.write(f"Usando nivel: **{nivel}** para generar rutina")
    
    # Inicializar equipos seleccionados si no existe
    if 'equipos_seleccionados' not in st.session_state:
        st.session_state.equipos_seleccionados = []
    
    st.markdown("---")
    
    # Multiselect con equipos
    equipo = st.multiselect(
        "Equipo disponible:",
        ["Pesas", "Mancuernas", "Barra", "Bandas elásticas", 
         "Máquinas", "Solo peso corporal"],
        default=st.session_state.equipos_seleccionados,
        help="Selecciona manualmente o usa el detector de imágenes arriba"
    )
    
    # Actualizar session_state
    st.session_state.equipos_seleccionados = equipo
    
    # Mostrar sugerencias si existen
    if 'equipos_sugeridos' in st.session_state and st.session_state.equipos_sugeridos:
        equipos_faltantes = [eq for eq in st.session_state.equipos_sugeridos 
                             if eq not in equipo]
        if equipos_faltantes:
            st.caption(f"💡 Detectados por CNN pero no agregados: {', '.join(equipos_faltantes)}")
    
    tiempo = st.slider("Minutos por sesión:", 20, 120, 60)
    
    st.markdown("---")
    generar = st.button("Generar Rutina", type="primary", use_container_width=True)

# Sistema de prompts
SYSTEM_PROMPT = """Eres GymMentor, un entrenador personal experto y motivador.
Diseñas rutinas de ejercicio personalizadas, claras y efectivas.
Incluyes:
- Ejercicios específicos con series y repeticiones
- Breve explicación de técnica
- Consejo motivacional al final
Sé conciso pero completo. Usa emojis ocasionalmente para hacerlo amigable."""

# Generar rutina
if generar:
    with st.spinner("Creando tu rutina personalizada..."):
        
        # Construir prompt del usuario
        equipo_texto = ', '.join(equipo) if equipo else 'peso corporal'
        user_prompt = f"""Genera una rutina de entrenamiento con estos datos:
- Días por semana: {dias}
- Objetivo: {objetivo}
- Nivel: {nivel}
- Equipo disponible: {equipo_texto}
- Tiempo por sesión: {tiempo} minutos

Estructura la rutina por días de la semana y sé específico con ejercicios, series y repeticiones."""
        
        try:
            # Llamar a la API
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            rutina = response.choices[0].message.content
            
            # Mostrar resultado
            st.success("¡Rutina generada!")
            st.markdown(rutina)
            
            # Guardar en session_state
            if 'rutinas' not in st.session_state:
                st.session_state.rutinas = []
            st.session_state.rutinas.append({
                'perfil': f"{dias}d - {objetivo} - {nivel}",
                'rutina': rutina
            })
            
        except Exception as e:
            st.error(f"Error al generar rutina: {e}")
            st.info("Verifica que tu API key esté correcta en el archivo .env")

# Mostrar historial
if 'rutinas' in st.session_state and st.session_state.rutinas:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Historial")
    for i, r in enumerate(st.session_state.rutinas[-3:]):
        with st.sidebar.expander(f"Rutina {i+1}: {r['perfil']}"):
            st.text(r['rutina'][:150] + "...")

# Predictor de Adherencia con LSTM
st.markdown("---")
st.subheader("Predictor de Consistencia (LSTM)")
st.write("Marca los días que entrenaste esta semana y predeciremos tu adherencia:")

col1, col2 = st.columns([2, 1])

with col1:
    # Checkboxes para cada día
    dias_nombres = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    dias_semana = []
    
    cols = st.columns(7)
    for i, dia in enumerate(dias_nombres):
        with cols[i]:
            entreno = st.checkbox(dia[:3], key=f"dia_{i}", help=dia)
            dias_semana.append(1 if entreno else 0)
    
    # Contador visual
    total_dias = sum(dias_semana)
    st.progress(total_dias / 7)
    st.caption(f"Entrenaste {total_dias} de 7 días esta semana")

with col2:
    st.write("")  # Espaciado
    st.write("")
    
    if st.button("Predecir Adherencia", type="primary", use_container_width=True):
        # Hacer predicción
        X = np.array(dias_semana).reshape(1, 7, 1)
        prob = lstm_model.predict(X, verbose=0)[0][0]
        
        # Mostrar métrica
        st.metric(
            label="Probabilidad de entrenar mañana",
            value=f"{prob*100:.1f}%",
            delta=None
        )
        
        # Feedback personalizado basado en probabilidad
        if prob > 0.7:
            st.success("¡Excelente consistencia! Sigue así, estás en racha.")
            consejo = "Mantén este ritmo. Tu cuerpo se está adaptando muy bien."
        elif prob > 0.5:
            st.info("Vas bien, pero puedes mejorar tu consistencia.")
            consejo = "Intenta no faltar más de 2 días seguidos para mantener el progreso."
        elif prob > 0.3:
            st.warning("Cuidado, estás perdiendo consistencia.")
            consejo = "Considera reducir el tiempo de entrenamiento pero mantener la frecuencia."
        else:
            st.error("Alto riesgo de abandono detectado.")
            consejo = "Es normal tener semanas difíciles. Empieza de nuevo mañana, aunque sean solo 20 minutos."
        
        with st.expander("💡 Consejo personalizado"):
            st.write(consejo)
        
        # Guardar en historial
        if 'historial_adherencia' not in st.session_state:
            st.session_state.historial_adherencia = []
        
        st.session_state.historial_adherencia.append({
            'semana': dias_semana,
            'probabilidad': prob
        })

# Mostrar tendencia si hay historial
if 'historial_adherencia' in st.session_state and len(st.session_state.historial_adherencia) > 1:
    st.markdown("---")
    st.subheader("Tu Tendencia de Adherencia")
    
    probs = [h['probabilidad'] * 100 for h in st.session_state.historial_adherencia]
    
    # Crear gráfico simple
    import pandas as pd
    df = pd.DataFrame({
        'Semana': [f"Semana {i+1}" for i in range(len(probs))],
        'Adherencia (%)': probs
    })
    
    st.line_chart(df.set_index('Semana'))
    
    # Análisis de tendencia
    if len(probs) >= 2:
        cambio = probs[-1] - probs[-2]
        if cambio > 5:
            st.success(f"¡Mejoraste {cambio:.1f}% vs semana anterior!")
        elif cambio < -5:
            st.warning(f"Bajaste {abs(cambio):.1f}% vs semana anterior")
        else:
            st.info("Te mantienes estable")

# Sistema de registro de entrenamientos
st.markdown("---")
st.subheader("Registro de Entrenamientos")

col1, col2, col3 = st.columns(3)

# Inicializar historial si no existe
if 'entrenamientos' not in st.session_state:
    st.session_state.entrenamientos = []

with col1:
    if st.button("Registrar entrenamiento de hoy", use_container_width=True):
        hoy = datetime.now().strftime("%Y-%m-%d")
        if hoy not in st.session_state.entrenamientos:
            st.session_state.entrenamientos.append(hoy)
            st.success("¡Entrenamiento registrado!")
            st.balloons()
        else:
            st.info("Ya registraste el entrenamiento de hoy")

with col2:
    total_entrenamientos = len(st.session_state.entrenamientos)
    st.metric("Total de entrenamientos", total_entrenamientos)

with col3:
    if total_entrenamientos > 0:
        # Calcular racha actual
        fechas = sorted([datetime.strptime(f, "%Y-%m-%d") for f in st.session_state.entrenamientos])
        racha = 1
        for i in range(len(fechas)-1, 0, -1):
            diff = (fechas[i] - fechas[i-1]).days
            if diff <= 2:  # Permite 1 día de descanso
                racha += 1
            else:
                break
        st.metric("Racha actual", f"{racha} días")
    else:
        st.metric("Racha actual", "0 días")

# Mostrar calendario de últimos 30 días
if total_entrenamientos > 0:
    st.markdown("### Actividad de los últimos 30 días")
    
    # Generar últimos 30 días
    hoy = datetime.now()
    ultimos_30 = [(hoy - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
    ultimos_30.reverse()
    
    # Crear DataFrame
    df_actividad = pd.DataFrame({
        'Fecha': ultimos_30,
        'Entrenó': [1 if fecha in st.session_state.entrenamientos else 0 for fecha in ultimos_30]
    })
    
    # Gráfico con plotly
    fig = px.bar(df_actividad, x='Fecha', y='Entrenó', 
                 color='Entrenó',
                 color_continuous_scale=['#FF4B4B', '#00CC00'],
                 labels={'Entrenó': 'Actividad'})
    fig.update_layout(showlegend=False, height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas adicionales
    col1, col2, col3, col4 = st.columns(4)
    
    entrenamientos_30d = sum([1 for fecha in st.session_state.entrenamientos 
                              if fecha in ultimos_30])
    
    with col1:
        st.metric("Últimos 30 días", f"{entrenamientos_30d} entrenamientos")
    with col2:
        porcentaje = (entrenamientos_30d / 30) * 100
        st.metric("Consistencia", f"{porcentaje:.0f}%")
    with col3:
        promedio_semanal = entrenamientos_30d / 4.3
        st.metric("Promedio semanal", f"{promedio_semanal:.1f} días")
    with col4:
        if entrenamientos_30d >= 20:
            nivel_com = "Elite"
        elif entrenamientos_30d >= 15:
            nivel_com = "Muy bien"
        elif entrenamientos_30d >= 10:
            nivel_com = "Bien"
        else:
            nivel_com = "Mejorable"
        st.metric("Nivel compromiso", nivel_com)

# Generador de motivación diaria
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Motivación del Día")

with col2:
    if st.button("Nueva frase", use_container_width=True):
        with st.spinner("Generando..."):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Eres un motivador fitness. Genera UNA frase motivacional corta y poderosa (máximo 20 palabras)."},
                        {"role": "user", "content": "Dame una frase motivacional para ir al gym hoy"}
                    ],
                    temperature=0.9,
                    max_tokens=50
                )
                frase = response.choices[0].message.content
                st.session_state.frase_motivacional = frase
            except:
                st.session_state.frase_motivacional = " El dolor de hoy es la fuerza de mañana."

if 'frase_motivacional' in st.session_state:
    st.info(st.session_state.frase_motivacional)
else:
    st.info("Click en 'Nueva frase' para obtener motivación personalizada")

# Sección de chat
st.markdown("---")
st.subheader("💬 Pregúntame sobre ejercicios o técnica")

# Inicializar historial de chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("Ej: ¿Cómo hacer sentadillas correctamente?"):
    # Agregar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *st.session_state.messages
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                respuesta = response.choices[0].message.content
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except Exception as e:
                st.error(f"Error: {e}")