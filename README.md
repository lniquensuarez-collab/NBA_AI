# 🏀 NBA ULTIMATE AI

Aplicación web cuantitativa desarrollada en **Python + Streamlit** enfocada en la predicción probabilística, simulación de Monte Carlo correlacionada y detección de valor esperado (`EV+`) en mercados de apuestas de la **NBA** (optimizada especialmente para **Betano** y **Betsafe**).

## 🚀 Características Principales
- **100% Libre de API Keys de Pago**: Ingesta histórica multihilo (`ThreadPoolExecutor`) y cartelera en vivo desde los endpoints JSON públicos de **ESPN NBA**.
- **Motor Matemático Ensemble de 3 Pilares (Zero Data Leakage)**:
  1. **Modelo Posesional y de Eficiencia (`L-BFGS-B` / Decaimiento Exponencial `half_life`)**: Ritmo esperado ($\text{Pace}_{\text{esp}}$), puntos en 48 minutos ($\mu_L, \mu_V$), 1er Cuarto (`Q1` ~25.2%) y 1ra Mitad (`1H` ~50.8%).
  2. **Sistema Elo NBA con Margen de Victoria (MOV FiveThirtyEight)**: Factor de desaceleración logarítmica, conversión directa a spread (`diff_elo / 28.0`) y regresión del `25%` hacia `1500` entre temporadas.
  3. **Four Factors de Dean Oliver & EWMA Híbrido (70% Local/Visita + 30% General)**: `efg_diff`, `tov_pct_diff`, `orb_pct_diff`, `ftr_diff`, `net_rtg_diff` y rachas de 5/10 partidos.
- **Factores Tácticos Exclusivos NBA (`ajuste_fatiga_nba` & `ALTITUD_NBA`)**:
  - Penalización automática por **Back-to-Back (`B2B`)**: `-2.6 pts` visitante y `-1.5 pts` local.
  - Penalización por densidad de calendario (**`3-in-4` o `4-in-6` noches**): `-1.2 pts`.
  - Bono de **Ventaja de Descanso (`Rest Advantage`)**: `+1.0 pt` ($\ge 3$ días vs $1$ día).
  - **Altitud de Pabellones**: **Denver Nuggets** (`1,609 msnm`, `+1.6 pts`) y **Utah Jazz** (`1,288 msnm`, `+1.1 pts`), intensificados un `+40%` si el visitante llega con fatiga acumulada.
- **Optimizaciones Betano / Betsafe**:
  - **⚡ Bono Matemático de Pago Anticipado (+20 pts)**: Cálculo mediante propiedades de puente browniano / caminata aleatoria de la probabilidad adicional de tocar `+20 puntos` de ventaja y perder el encuentro.
  - **💹 Calculadora EV% y Kelly 1/4 Conservador**: Con desmargenado por Método de Potencia (*Power Method*) y evaluación en el percentil 25 (`P25`) de Monte Carlo.
- **🎯 Pestaña Dedicada de Combinadas + 🌶️ Zona Picante (Cuota 5.00+)**:
  - Buscador automático de la cartelera NBA del día, Ranking de Seguridad, **🥇 Mejor Combinada Doble**, **🥈 Mejor Combinada Triple**, **🌶️ Escalera Picante** y **💣 Bomba Express**.

## 🛠️ Ejecución Local y Pruebas

```bash
pip install -r requirements.txt
python -m tests.test_engine
python -m tests.test_app_smoke
streamlit run app.py
```
