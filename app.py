"""
🏀 NBA ULTIMATE AI
Aplicación Web Cuantitativa de Predicción Probabilística y Detección de Valor (EV+)
en mercados de apuestas de la NBA (Optimizada para Betano y Betsafe).
"""

from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from engine.config import (
    NBA_TEAMS,
    DEFAULT_HALF_LIFE,
    DEFAULT_MC_SIMS,
    DEFAULT_SEASONS,
    ALTITUD_NBA,
    normalizar_equipo,
)
from engine.db import (
    init_db,
    cargar_partidos_df,
    desmargenar_cuotas_power,
)
from engine.data_bot import (
    descargar_historial_nba,
    actualizar_partidos_recientes,
    buscar_cartelera_nba,
    importar_archivo_usuario,
    generar_partidos_sinteticos_nba,
)
from engine.features import (
    construir_dataset_cronologico,
    generar_tabla_power_ranking,
)
from engine.models import (
    NBAEnsembleEngine,
    calcular_ev_y_kelly,
    armar_combinadas_y_zona_picante,
)
from engine.backtest import ejecutar_backtest_walk_forward
from engine.players import (
    obtener_jugadores_equipo,
    obtener_bajas_detectadas_equipo,
    obtener_catalogo_completo_jugadores,
    calcular_impacto_bajas,
    proyectar_y_simular_jugador,
    proyectar_plantilla_partido,
    sincronizar_jugadores_espn,
)


def configurar_pagina() -> None:
    st.set_page_config(
        page_title="🏀 NBA ULTIMATE AI | Quant & EV+ Engine",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _entrenar_motor_base(
    half_life: float, db_path: Optional[str] = None
) -> tuple[NBAEnsembleEngine, pd.DataFrame, pd.DataFrame]:
    """
    Inicializa SQLite, verifica que haya historial cargado (si está vacío, inicializa un
    dataset semilla NBA para que la app sea inmediatamente operativa en producción/cloud)
    y entrena los 3 pilares del motor Ensemble.
    """
    init_db(db_path)
    df = cargar_partidos_df(db_path=db_path)
    if df.empty or len(df) < 360:
        generar_partidos_sinteticos_nba(n_partidos=480, temporadas=[2025, 2026], db_path=db_path)
        df = cargar_partidos_df(db_path=db_path)

    df_feat, tracker, elo_sys = construir_dataset_cronologico(df)
    engine = NBAEnsembleEngine(half_life=half_life, n_sims=DEFAULT_MC_SIMS)
    engine.tracker = tracker
    engine.elo_sys = elo_sys
    engine.fit(df, df_features_precomputed=df_feat)
    return engine, df, df_feat


@st.cache_resource(show_spinner=False)
def _cargar_motor_cached(
    half_life: float, db_path: Optional[str] = None
) -> tuple[NBAEnsembleEngine, pd.DataFrame, pd.DataFrame]:
    return _entrenar_motor_base(half_life=half_life, db_path=db_path)


def cargar_motor_entrenado(
    half_life: float, n_sims: int, db_path: Optional[str] = None
) -> tuple[NBAEnsembleEngine, pd.DataFrame, pd.DataFrame]:
    if hasattr(st, "cache_resource") and db_path is None:
        engine, df, df_feat = _cargar_motor_cached(half_life=half_life, db_path=db_path)
    else:
        engine, df, df_feat = _entrenar_motor_base(half_life=half_life, db_path=db_path)
    engine.n_sims = int(n_sims)
    return engine, df, df_feat


@st.cache_data(show_spinner=False)
def _ejecutar_backtest_cached(
    n_rows: int,
    max_fecha: str,
    warmup_games: int,
    half_life: float,
    umbral_ev_pct: float,
    _df_partidos: pd.DataFrame,
) -> Dict[str, Any]:
    return ejecutar_backtest_walk_forward(
        df_partidos=_df_partidos,
        warmup_games=warmup_games,
        half_life=half_life,
        umbral_ev_pct=umbral_ev_pct,
    )


def render_sidebar() -> Dict[str, Any]:
    """Renderiza la barra lateral con todos los parámetros cuantitativos y botón de Combinadas."""
    st.sidebar.markdown("## 🏀 NBA ULTIMATE AI")
    st.sidebar.caption("Motor Cuantitativo de 3 Pilares · Monte Carlo · EV+ Betano & Betsafe")

    casa_apuestas = st.sidebar.selectbox(
        "🏦 Casa de apuestas",
        options=[
            "Betano (Pago Anticipado +20 pts / SuperCuotas)",
            "Betsafe (Perú / Bet Builder)",
            "Estándar (Final del partido)",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Parámetros del Modelo")
    half_life = st.sidebar.slider(
        "⏳ Decaimiento Temporal (Half-Life días)",
        min_value=30,
        max_value=240,
        value=int(DEFAULT_HALF_LIFE),
        step=15,
        help="Controla el peso exponencial de los partidos recientes frente a los antiguos.",
    )
    n_sims = st.sidebar.slider(
        "🎲 Simulaciones Monte Carlo",
        min_value=5000,
        max_value=10000,
        value=int(DEFAULT_MC_SIMS),
        step=500,
        help="Cantidad de partidos simulados con incertidumbre epistémica correlacionada.",
    )
    banca_actual = st.sidebar.number_input(
        "💰 Banca Total (Unidades / S/ / USD)",
        min_value=10.0,
        max_value=1000000.0,
        value=1000.0,
        step=50.0,
    )

    st.sidebar.markdown("---")
    ir_a_combinadas = st.sidebar.button(
        "🎯 Combinadas",
        use_container_width=True,
        help="Ir directamente al constructor de Combinadas Seguras y Zona Picante 5.0+",
    )
    if ir_a_combinadas:
        st.session_state["foco_combinadas"] = True

    return {
        "casa_apuestas": casa_apuestas,
        "half_life": float(half_life),
        "n_sims": int(n_sims),
        "banca": float(banca_actual),
        "ir_a_combinadas": ir_a_combinadas,
    }


def _generar_opciones_fechas(dias_atras: int = 10, dias_adelante: int = 14) -> List[str]:
    hoy = datetime.now().date()
    return [
        (hoy + timedelta(days=d)).strftime("%Y-%m-%d")
        for d in range(-dias_atras, dias_adelante + 1)
    ]


def render_tab_prediccion(
    engine: NBAEnsembleEngine, params: Dict[str, Any]
) -> None:
    """Pestaña 1: 🔮 Predicción Individual y Mercados Completos (Moneyline, Spread, Totales, Q1/1H)."""
    st.subheader("🔮 Predicción Probabilística & Simulación de Monte Carlo")
    teams_sorted = sorted(list(NBA_TEAMS.keys()))
    fechas_disp = _generar_opciones_fechas(dias_atras=10, dias_adelante=14)
    hoy_iso = datetime.now().strftime("%Y-%m-%d")
    idx_hoy = fechas_disp.index(hoy_iso) if hoy_iso in fechas_disp else 0

    col1, col2, col3 = st.columns([2, 2, 1.3])
    with col1:
        default_local = teams_sorted.index("Boston Celtics") if "Boston Celtics" in teams_sorted else 0
        local = st.selectbox("🏠 Equipo Local (Home)", options=teams_sorted, index=default_local, key="pred_local")
    with col2:
        default_visita = teams_sorted.index("Los Angeles Lakers") if "Los Angeles Lakers" in teams_sorted else 1
        visita = st.selectbox("✈️ Equipo Visitante (Away)", options=teams_sorted, index=default_visita, key="pred_visita")
    with col3:
        fecha_sel = st.selectbox("📅 Fecha del Partido", options=fechas_disp, index=idx_hoy, key="pred_fecha")

    if local == visita:
        st.warning("Selecciona dos franquicias NBA distintas para ejecutar la simulación.")
        return

    # Selector de Bajas / Ausencias de Estrellas para ajustar el Spread y Total del partido en vivo
    roster_l = obtener_jugadores_equipo(local)
    roster_v = obtener_jugadores_equipo(visita)
    nombres_l = [p["jugador"] for p in roster_l]
    nombres_v = [p["jugador"] for p in roster_v]
    bajas_auto_l = [b for b in obtener_bajas_detectadas_equipo(local) if b in nombres_l]
    bajas_auto_v = [b for b in obtener_bajas_detectadas_equipo(visita) if b in nombres_v]

    with st.expander("🏥 Simulador de Bajas / Ausencias de Estrellas y Resumen del Quinteto (ESPN Injuries)", expanded=False):
        cb1, cb2 = st.columns(2)
        with cb1:
            bajas_l = st.multiselect(
                f"Bajas / Ausentes en {local}",
                options=nombres_l,
                default=bajas_auto_l,
                key=f"bajas_local_pred_{local}",
            )
        with cb2:
            bajas_v = st.multiselect(
                f"Bajas / Ausentes en {visita}",
                options=nombres_v,
                default=bajas_auto_v,
                key=f"bajas_visita_pred_{visita}",
            )

        info_bajas = calcular_impacto_bajas(local, visita, bajas_local=bajas_l, bajas_visita=bajas_v)
        if bajas_l or bajas_v:
            st.warning(
                f"⚠️ **Ajuste por Bajas Aplicado:** Impacto en Spread Local: `{info_bajas['ajuste_spread_local_pts']:+.2f} pts` "
                f"| Impacto en Total del Partido: `{info_bajas['ajuste_total_pts']:+.2f} pts` "
                f"(Penalización {local}: `{info_bajas['penalizacion_local_pts']:.1f} pts`, "
                f"Penalización {visita}: `{info_bajas['penalizacion_visita_pts']:.1f} pts`)."
            )
        else:
            st.caption("Ambos equipos con quintetos estelares completos (0.00 pts de ajuste por bajas).")

    pred_key = (
        f"{local}_{visita}_{fecha_sel}_{params['casa_apuestas']}_"
        f"{params['half_life']}_{params['n_sims']}_{tuple(bajas_l)}_{tuple(bajas_v)}"
    )
    if st.session_state.get("ultima_pred_key") != pred_key or "ultima_prediccion" not in st.session_state:
        st.session_state["ultima_prediccion"] = engine.predecir_partido_completo(
            local=local,
            visita=visita,
            fecha=fecha_sel,
            casa_apuestas=params["casa_apuestas"],
            n_sims=params["n_sims"],
            ajuste_bajas_spread_local=info_bajas["ajuste_spread_local_pts"],
            ajuste_bajas_total=info_bajas["ajuste_total_pts"],
        )
        st.session_state["ultima_pred_key"] = pred_key
        st.session_state["ultimas_bajas_l"] = bajas_l
        st.session_state["ultimas_bajas_v"] = bajas_v
    pred = st.session_state["ultima_prediccion"]

    # 1. Tarjeta obligatoria con el diagnóstico físico (Fatiga B2B, 3-in-4, Rest Advantage y Altitud)
    fatiga = pred["fatiga"]
    if fatiga["hay_alerta"]:
        st.warning(fatiga["tarjeta_diagnostico"])
    else:
        st.info(fatiga["tarjeta_diagnostico"])

    # 2. Semáforo de Consenso (Detector de Partidos Trampa)
    consenso = pred["consenso"]
    if consenso["es_partido_trampa"]:
        st.error(f"**{consenso['estado']}** — {consenso['detalle']}")
    elif consenso["nivel"] == "AMARILLO":
        st.warning(f"**{consenso['estado']}** — {consenso['detalle']}")
    else:
        st.success(f"**{consenso['estado']}** — {consenso['detalle']}")

    # 3. Métricas Principales del Partido
    mc = pred["monte_carlo"]
    pa = pred["pago_anticipado"]
    pilares = pred["pilares"]
    usa_pa = ("betano" in params["casa_apuestas"].lower()) or ("betsafe" in params["casa_apuestas"].lower())

    m1, m2, m3, m4 = st.columns(4)
    p_l_show = pa["p_efectiva_local"] if usa_pa else mc["moneyline"]["prob_local"]
    p_v_show = pa["p_efectiva_visita"] if usa_pa else mc["moneyline"]["prob_visita"]

    m1.metric(
        f"🏠 Gana {local}",
        f"{p_l_show * 100:.1f}%",
        f"P10–P90: {mc['moneyline']['p10_local']*100:.0f}%–{mc['moneyline']['p90_local']*100:.0f}% | Cuota {1/p_l_show:.2f}",
    )
    m2.metric(
        f"✈️ Gana {visita}",
        f"{p_v_show * 100:.1f}%",
        f"P10–P90: {mc['moneyline']['p10_visita']*100:.0f}%–{mc['moneyline']['p90_visita']*100:.0f}% | Cuota {1/p_v_show:.2f}",
    )
    m3.metric(
        "📏 Spread Justo del Modelo",
        mc["spread"]["etiqueta_spread_justo"],
        f"Marcador Esp.: {local} {pilares['mu_local']:.1f} - {pilares['mu_visita']:.1f} {visita}",
    )
    m4.metric(
        "🔥 Total Esperado (48m)",
        f"{mc['totales']['total_esperado']:.1f} pts",
        f"Ritmo (Pace): {pilares['pace_esperado']:.1f} poss | OT: {mc['moneyline']['prob_prorroga']*100:.1f}%",
    )

    # 4. Bono Matemático de Pago Anticipado +20 pts (Betano / Betsafe)
    st.markdown("#### ⚡ Bono Matemático de Pago Anticipado NBA (+20 Puntos en Betano / Betsafe)")
    c_pa1, c_pa2 = st.columns(2)
    with c_pa1:
        st.markdown(
            f"**🏠 {local}:** Prob. Final Base `{pa['p_ml_local_base']*100:.2f}%` "
            f"+ Bono Remontada (+20 pts) **`+{pa['bono_20pts_local']*100:.2f}%`** "
            f"→ **Prob. Efectiva PA: `{pa['p_efectiva_local']*100:.2f}%`** "
            f"(Cuota Justa mejora de `{pa['cuota_justa_base_local']:.2f}` a **`{pa['cuota_justa_pa_local']:.2f}`**)"
        )
    with c_pa2:
        st.markdown(
            f"**✈️ {visita}:** Prob. Final Base `{pa['p_ml_visita_base']*100:.2f}%` "
            f"+ Bono Remontada (+20 pts) **`+{pa['bono_20pts_visita']*100:.2f}%`** "
            f"→ **Prob. Efectiva PA: `{pa['p_efectiva_visita']*100:.2f}%`** "
            f"(Cuota Justa mejora de `{pa['cuota_justa_base_visita']:.2f}` a **`{pa['cuota_justa_pa_visita']:.2f}`**)"
        )

    # 5. Resumen: Top 2-3 Opciones de Bajo Riesgo por Partido
    st.markdown("#### 🛡️ Top Opciones Inteligentes de Bajo Riesgo (Zona Óptima 66%–88.5%)")
    cols_br = st.columns(len(pred["bajo_riesgo"]))
    for idx_br, item_br in enumerate(pred["bajo_riesgo"]):
        with cols_br[idx_br]:
            st.markdown(
                f"**{item_br['categoria']}**\n\n"
                f"🎯 **{item_br['mercado']}**\n"
                f"- Probabilidad: **`{item_br['probabilidad']*100:.1f}%`** (Piso P10: `{item_br['piso_p10']*100:.1f}%`)\n"
                f"- Cuota Justa: **`{item_br['cuota_justa']:.2f}`**\n"
                f"- _{item_br['justificacion']}_"
            )

    # 6. Desglose de los 3 Pilares y Four Factors
    with st.expander("🧠 Ver Desglose de los 3 Pilares Matemáticos y Four Factors (EWMA 70/30)", expanded=False):
        df_pilares = pd.DataFrame(
            [
                {
                    "Pilar del Modelo": "1. Posesional & Eficiencia (L-BFGS-B / Half-Life)",
                    f"Prob. {local} %": f"{pilares['p_posesional']*100:.1f}%",
                    f"Prob. {visita} %": f"{(1-pilares['p_posesional'])*100:.1f}%",
                    "Ventaja Local (pts)": f"{pilares['spread_posesional']:+.2f}",
                },
                {
                    "Pilar del Modelo": "2. Sistema Elo NBA (MOV FiveThirtyEight)",
                    f"Prob. {local} %": f"{pilares['p_elo']*100:.1f}%",
                    f"Prob. {visita} %": f"{(1-pilares['p_elo'])*100:.1f}%",
                    "Ventaja Local (pts)": f"{pilares['spread_elo']:+.2f}",
                },
                {
                    "Pilar del Modelo": "3. Four Factors & Forma EWMA Híbrida (70/30)",
                    f"Prob. {local} %": f"{pilares['p_four_factors_ml']*100:.1f}%",
                    f"Prob. {visita} %": f"{(1-pilares['p_four_factors_ml'])*100:.1f}%",
                    "Ventaja Local (pts)": f"{pilares['spread_four_factors_ml']:+.2f}",
                },
            ]
        )
        st.dataframe(df_pilares, use_container_width=True, hide_index=True)

        ft = pred["features"]
        df_ff = pd.DataFrame(
            [
                {
                    "Equipo": local,
                    "Elo": round(ft["elo_l"], 1),
                    "ORtg (70/30)": round(ft["ortg_l"], 1),
                    "DRtg (70/30)": round(ft["drtg_l"], 1),
                    "Net Rating": round(ft["net_rtg_l"], 2),
                    "Pace": round(ft["pace_l"], 1),
                    "eFG%": f"{ft['efg_l']*100:.1f}%",
                    "TOV%": f"{ft['tov_pct_l']*100:.1f}%",
                    "ORB%": f"{ft['orb_pct_l']*100:.1f}%",
                    "FTR": f"{ft['ftr_l']*100:.1f}%",
                },
                {
                    "Equipo": visita,
                    "Elo": round(ft["elo_v"], 1),
                    "ORtg (70/30)": round(ft["ortg_v"], 1),
                    "DRtg (70/30)": round(ft["drtg_v"], 1),
                    "Net Rating": round(ft["net_rtg_v"], 2),
                    "Pace": round(ft["pace_v"], 1),
                    "eFG%": f"{ft['efg_v']*100:.1f}%",
                    "TOV%": f"{ft['tov_pct_v']*100:.1f}%",
                    "ORB%": f"{ft['orb_pct_v']*100:.1f}%",
                    "FTR": f"{ft['ftr_v']*100:.1f}%",
                },
            ]
        )
        st.dataframe(df_ff, use_container_width=True, hide_index=True)

    # 7. Resumen de Estrellas y Top Player Props del Partido Actual (PTS, REB, AST, 3PM, PRA)
    with st.expander("⛹️‍♂️ Resumen de Estrellas del Partido & Top Player Props (PTS, REB, AST, 3PM, PRA)", expanded=False):
        ft_now = pred["features"]
        plantilla_proj = proyectar_plantilla_partido(
            equipo_local=local,
            equipo_visita=visita,
            pace_esperado=pilares["pace_esperado"],
            drtg_local=ft_now["drtg_l"],
            drtg_visita=ft_now["drtg_v"],
            b2b_local=bool(ft_now["b2b_l"]),
            b2b_visita=bool(ft_now["b2b_v"]),
            bajas_local=st.session_state.get("ultimas_bajas_l", []),
            bajas_visita=st.session_state.get("ultimas_bajas_v", []),
            n_sims=3000,
        )
        st.markdown("**🔥 Top 8 Player Props Más Seguros del Partido (Monte Carlo):**")
        st.dataframe(pd.DataFrame(plantilla_proj["top_props"]), use_container_width=True, hide_index=True)

    # 8. Tablas de Mercados: Spread / Hándicap, Totales Over/Under, Team Totals y Q1 / 1H
    sub1, sub2, sub3, sub4 = st.tabs(
        [
            "📏 Hándicap / Spread (-12.5 a +12.5)",
            "🔢 Totales Over / Under",
            "🏀 Totales por Equipo (Team Totals)",
            "⏱️ 1er Cuarto (Q1) y 1ra Mitad (1H)",
        ]
    )

    with sub1:
        df_sp = pd.DataFrame(mc["spread"]["tabla"])
        df_sp_view = pd.DataFrame(
            {
                "Línea": df_sp["linea_str"],
                f"Mercado {local}": df_sp["local_mercado"],
                f"Cubrir {local} %": (df_sp["prob_local"] * 100).round(1),
                f"Piso P10 {local} %": (df_sp["p10_local"] * 100).round(1),
                f"Cuota Justa {local}": df_sp["cuota_justa_local"],
                f"Mercado {visita}": df_sp["visita_mercado"],
                f"Cubrir {visita} %": (df_sp["prob_visita"] * 100).round(1),
                f"Piso P10 {visita} %": (df_sp["p10_visita"] * 100).round(1),
                f"Cuota Justa {visita}": df_sp["cuota_justa_visita"],
            }
        )
        st.dataframe(df_sp_view, use_container_width=True, hide_index=True)

    with sub2:
        df_tot = pd.DataFrame(mc["totales"]["tabla"])
        df_tot_view = pd.DataFrame(
            {
                "Línea de Puntos": df_tot["linea"],
                "Over (Más de) %": (df_tot["prob_over"] * 100).round(1),
                "Piso P10 Over %": (df_tot["p10_over"] * 100).round(1),
                "Cuota Justa Over": df_tot["cuota_justa_over"],
                "Under (Menos de) %": (df_tot["prob_under"] * 100).round(1),
                "Piso P10 Under %": (df_tot["p10_under"] * 100).round(1),
                "Cuota Justa Under": df_tot["cuota_justa_under"],
            }
        )
        st.dataframe(df_tot_view, use_container_width=True, hide_index=True)

    with sub3:
        ct1, ct2 = st.columns(2)
        with ct1:
            st.markdown(f"**🏠 Total Individual: {local} (Esp: {pilares['mu_local']:.1f} pts)**")
            df_ttl = pd.DataFrame(mc["team_totals"]["local"])
            st.dataframe(
                pd.DataFrame(
                    {
                        "Línea": df_ttl["linea"],
                        "Over %": (df_ttl["prob_over"] * 100).round(1),
                        "Cuota Over": df_ttl["cuota_justa_over"],
                        "Under %": (df_ttl["prob_under"] * 100).round(1),
                        "Cuota Under": df_ttl["cuota_justa_under"],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with ct2:
            st.markdown(f"**✈️ Total Individual: {visita} (Esp: {pilares['mu_visita']:.1f} pts)**")
            df_ttv = pd.DataFrame(mc["team_totals"]["visita"])
            st.dataframe(
                pd.DataFrame(
                    {
                        "Línea": df_ttv["linea"],
                        "Over %": (df_ttv["prob_over"] * 100).round(1),
                        "Cuota Over": df_ttv["cuota_justa_over"],
                        "Under %": (df_ttv["prob_under"] * 100).round(1),
                        "Cuota Under": df_ttv["cuota_justa_under"],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with sub4:
        q1_d = mc["q1"]
        h1_d = mc["h1"]
        cq1, ch1 = st.columns(2)
        with cq1:
            st.markdown("##### ⏱️ 1er Cuarto (`Q1` · ~25.2% del Total)")
            st.write(
                f"- **Marcador Esperado Q1:** `{local} {q1_d['mu_local']:.1f} - {q1_d['mu_visita']:.1f} {visita}` "
                f"(Total Q1: `{q1_d['total_esperado']:.1f} pts`)"
            )
            st.write(
                f"- **Ganador Q1:** `{local} {q1_d['prob_gana_local']*100:.1f}%` (Cuota `{q1_d['cuota_local']:.2f}`) vs "
                f"`{visita} {q1_d['prob_gana_visita']*100:.1f}%` (Cuota `{q1_d['cuota_visita']:.2f}`)"
            )
            st.write(
                f"- **Hándicap Q1:** `{local} {q1_d['linea_handicap_local']:+.1f}` → Cubre `{q1_d['prob_cubre_local']*100:.1f}%`"
            )
            st.write(
                f"- **Total Q1 (`{q1_d['linea_total']:.1f}`):** Over `{q1_d['prob_over']*100:.1f}%` "
                f"(`{q1_d['cuota_over']:.2f}`) | Under `{q1_d['prob_under']*100:.1f}%` (`{q1_d['cuota_under']:.2f}`)"
            )
        with ch1:
            st.markdown("##### ⏱️ 1ra Mitad / Descanso (`1H` · ~50.8% del Total)")
            st.write(
                f"- **Marcador Esperado 1H:** `{local} {h1_d['mu_local']:.1f} - {h1_d['mu_visita']:.1f} {visita}` "
                f"(Total 1H: `{h1_d['total_esperado']:.1f} pts`)"
            )
            st.write(
                f"- **Ganador 1H:** `{local} {h1_d['prob_gana_local']*100:.1f}%` (Cuota `{h1_d['cuota_local']:.2f}`) vs "
                f"`{visita} {h1_d['prob_gana_visita']*100:.1f}%` (Cuota `{h1_d['cuota_visita']:.2f}`)"
            )
            st.write(
                f"- **Hándicap 1H:** `{local} {h1_d['linea_handicap_local']:+.1f}` → Cubre `{h1_d['prob_cubre_local']*100:.1f}%`"
            )
            st.write(
                f"- **Total 1H (`{h1_d['linea_total']:.1f}`):** Over `{h1_d['prob_over']*100:.1f}%` "
                f"(`{h1_d['cuota_over']:.2f}`) | Under `{h1_d['prob_under']*100:.1f}%` (`{h1_d['cuota_under']:.2f}`)"
            )


def render_tab_ev_kelly(
    engine: NBAEnsembleEngine, params: Dict[str, Any]
) -> None:
    """Pestaña 2: 💹 Calculadora de Valor Esperado (EV%) y Gestión de Banca (Kelly 1/4 Conservador)."""
    st.subheader("💹 Calculadora Cuantitativa de Valor Esperado (EV+) & Gestión de Banca (1/4 Kelly)")

    pred = st.session_state.get("ultima_prediccion")
    if not pred:
        pred = engine.predecir_partido_completo(
            local="Boston Celtics",
            visita="Los Angeles Lakers",
            casa_apuestas=params["casa_apuestas"],
            n_sims=params["n_sims"],
        )

    local = pred["local"]
    visita = pred["visita"]
    mc = pred["monte_carlo"]
    pa = pred["pago_anticipado"]
    usa_pa = ("betano" in params["casa_apuestas"].lower()) or ("betsafe" in params["casa_apuestas"].lower())

    st.caption(
        f"Partido activo: **{visita} @ {local}** | Casa seleccionada: **{params['casa_apuestas']}** | "
        f"Banca configurada: **{params['banca']:,.2f}**"
    )

    col_ml, col_sp, col_tot = st.columns(3)

    # 1. Mercado Moneyline (con soporte de Pago Anticipado +20 pts Betano/Betsafe)
    with col_ml:
        st.markdown("#### 🏆 1. Moneyline (Ganador)")
        cuota_ml_l = st.number_input(
            f"Cuota Local ({local})",
            min_value=1.01,
            max_value=50.0,
            value=float(round(pa["cuota_justa_pa_local"] * 1.04, 2)),
            step=0.05,
            key="ev_cuota_ml_l",
        )
        cuota_ml_v = st.number_input(
            f"Cuota Visita ({visita})",
            min_value=1.01,
            max_value=50.0,
            value=float(round(pa["cuota_justa_pa_visita"] * 1.04, 2)),
            step=0.05,
            key="ev_cuota_ml_v",
        )

        p_l_use = pa["p_efectiva_local"] if usa_pa else mc["moneyline"]["prob_local"]
        p_v_use = pa["p_efectiva_visita"] if usa_pa else mc["moneyline"]["prob_visita"]
        p25_l = max(mc["moneyline"]["p25_local"], p_l_use - 0.03)
        p25_v = max(mc["moneyline"]["p25_visita"], p_v_use - 0.03)

        ev_l = calcular_ev_y_kelly(p_l_use, cuota_ml_l, prob_p25=p25_l)
        ev_v = calcular_ev_y_kelly(p_v_use, cuota_ml_v, prob_p25=p25_v)
        p_fair_mkt_l, p_fair_mkt_v, overround_ml = desmargenar_cuotas_power(cuota_ml_l, cuota_ml_v)

        st.caption(f"Margen Casa (Power Method): `{overround_ml:.2f}%`")
        for eq_lbl, ev_obj in [(local, ev_l), (visita, ev_v)]:
            monto = (ev_obj["stake_kelly_pct"] / 100.0) * params["banca"]
            st.markdown(
                f"**{eq_lbl}** (`Cuota Justa: {ev_obj['cuota_justa']:.2f}`)\n"
                f"- **EV Central:** `{ev_obj['ev_pct']:+.2f}%` | **EV Conservador (P25):** `{ev_obj['ev_conservador_pct']:+.2f}%`\n"
                f"- **Stake 1/4 Kelly (máx 5%):** `{ev_obj['stake_kelly_pct']:.2f}%` (**`{monto:,.2f}`**)\n"
                f"- {ev_obj['recomendacion']}"
            )

    # 2. Mercado Spread / Hándicap
    with col_sp:
        st.markdown("#### 📏 2. Spread / Hándicap")
        lineas_disp = [r["linea"] for r in mc["spread"]["tabla"]]
        default_sp_idx = len(lineas_disp) // 2
        linea_sp_sel = st.selectbox(
            f"Línea de Hándicap para {local}",
            options=lineas_disp,
            index=default_sp_idx,
            format_func=lambda x: f"{local} {x:+.1f} / {visita} {-x:+.1f}",
            key="ev_linea_sp",
        )
        row_sp = next(r for r in mc["spread"]["tabla"] if r["linea"] == linea_sp_sel)

        cuota_sp_l = st.number_input(
            f"Cuota {local} {linea_sp_sel:+.1f}",
            min_value=1.01,
            max_value=25.0,
            value=1.91,
            step=0.02,
            key="ev_cuota_sp_l",
        )
        cuota_sp_v = st.number_input(
            f"Cuota {visita} {-linea_sp_sel:+.1f}",
            min_value=1.01,
            max_value=25.0,
            value=1.91,
            step=0.02,
            key="ev_cuota_sp_v",
        )

        ev_sp_l = calcular_ev_y_kelly(row_sp["prob_local"], cuota_sp_l, prob_p25=row_sp["p25_local"])
        ev_sp_v = calcular_ev_y_kelly(row_sp["prob_visita"], cuota_sp_v, prob_p25=row_sp["p25_visita"])

        for mkt_lbl, ev_obj in [
            (f"{local} {linea_sp_sel:+.1f}", ev_sp_l),
            (f"{visita} {-linea_sp_sel:+.1f}", ev_sp_v),
        ]:
            monto = (ev_obj["stake_kelly_pct"] / 100.0) * params["banca"]
            st.markdown(
                f"**{mkt_lbl}** (`Cuota Justa: {ev_obj['cuota_justa']:.2f}`)\n"
                f"- **EV Central:** `{ev_obj['ev_pct']:+.2f}%` | **EV Conservador (P25):** `{ev_obj['ev_conservador_pct']:+.2f}%`\n"
                f"- **Stake 1/4 Kelly:** `{ev_obj['stake_kelly_pct']:.2f}%` (**`{monto:,.2f}`**)\n"
                f"- {ev_obj['recomendacion']}"
            )

    # 3. Mercado Total de Puntos (Over / Under)
    with col_tot:
        st.markdown("#### 🔢 3. Total de Puntos (O/U)")
        lineas_tot_disp = [r["linea"] for r in mc["totales"]["tabla"]]
        linea_tot_sel = st.selectbox(
            "Línea de Puntos Totales",
            options=lineas_tot_disp,
            index=len(lineas_tot_disp) // 2,
            format_func=lambda x: f"{x:.1f} puntos",
            key="ev_linea_tot",
        )
        row_tot = next(r for r in mc["totales"]["tabla"] if r["linea"] == linea_tot_sel)

        cuota_ov = st.number_input(
            f"Cuota Over (Más de {linea_tot_sel:.1f})",
            min_value=1.01,
            max_value=25.0,
            value=1.91,
            step=0.02,
            key="ev_cuota_ov",
        )
        cuota_un = st.number_input(
            f"Cuota Under (Menos de {linea_tot_sel:.1f})",
            min_value=1.01,
            max_value=25.0,
            value=1.91,
            step=0.02,
            key="ev_cuota_un",
        )

        ev_ov = calcular_ev_y_kelly(row_tot["prob_over"], cuota_ov, prob_p25=row_tot["p25_over"])
        ev_un = calcular_ev_y_kelly(row_tot["prob_under"], cuota_un, prob_p25=row_tot["p25_under"])

        for mkt_lbl, ev_obj in [
            (f"Over {linea_tot_sel:.1f} pts", ev_ov),
            (f"Under {linea_tot_sel:.1f} pts", ev_un),
        ]:
            monto = (ev_obj["stake_kelly_pct"] / 100.0) * params["banca"]
            st.markdown(
                f"**{mkt_lbl}** (`Cuota Justa: {ev_obj['cuota_justa']:.2f}`)\n"
                f"- **EV Central:** `{ev_obj['ev_pct']:+.2f}%` | **EV Conservador (P25):** `{ev_obj['ev_conservador_pct']:+.2f}%`\n"
                f"- **Stake 1/4 Kelly:** `{ev_obj['stake_kelly_pct']:.2f}%` (**`{monto:,.2f}`**)\n"
                f"- {ev_obj['recomendacion']}"
            )


def render_tab_combinadas(
    engine: NBAEnsembleEngine, params: Dict[str, Any]
) -> None:
    """
    Pestaña 3: 🎯 Combinadas + Buscador Automático de la Cartelera NBA del Día
    + Armador de Combinadas Seguras (Doble/Triple) + 🌶️ Zona Picante (Cuota 5.00+).
    """
    st.subheader("🎯 Escáner de la Noche NBA, Combinadas Seguras & 🌶️ Zona Picante (Cuota 5.00+)")

    # Inicializar cartelera por defecto en session_state
    if "cartelera_activa" not in st.session_state or not st.session_state["cartelera_activa"]:
        hoy_str = datetime.now().strftime("%Y-%m-%d")
        st.session_state["cartelera_activa"] = [
            {"fecha": hoy_str, "local": "Boston Celtics", "visita": "New York Knicks"},
            {"fecha": hoy_str, "local": "Denver Nuggets", "visita": "Los Angeles Lakers"},
            {"fecha": hoy_str, "local": "Oklahoma City Thunder", "visita": "Dallas Mavericks"},
            {"fecha": hoy_str, "local": "Cleveland Cavaliers", "visita": "Milwaukee Bucks"},
        ]

    # 1. Buscador Automático de la Cartelera NBA del Día
    st.markdown("#### 🔎 1. Buscador Automático de la Cartelera NBA del Día (ESPN Live)")
    fechas_disp = _generar_opciones_fechas(dias_atras=10, dias_adelante=14)
    hoy_iso = datetime.now().strftime("%Y-%m-%d")
    idx_hoy = fechas_disp.index(hoy_iso) if hoy_iso in fechas_disp else 0

    b1, b2, b3 = st.columns([1.5, 1.5, 1.5])
    with b1:
        fecha_busqueda = st.selectbox("📅 Fecha de Cartelera", options=fechas_disp, index=idx_hoy, key="comb_fecha")
    with b2:
        rango_txt = st.selectbox(
            "📆 Rango de Búsqueda",
            options=["Solo hoy", "Hoy + 1 día", "Hoy + 2 días", "Hoy + 3 días"],
            index=0,
            key="comb_rango",
        )
        dias_adelante = {"Solo hoy": 0, "Hoy + 1 día": 1, "Hoy + 2 días": 2, "Hoy + 3 días": 3}[rango_txt]
    with b3:
        st.write("")
        st.write("")
        if st.button("🔎 Buscar Partidos en ESPN NBA", use_container_width=True):
            encontrados = buscar_cartelera_nba(
                fecha_inicio=fecha_busqueda,
                dias_adelante=dias_adelante,
                buscar_proxima_disponible=True,
            )
            if encontrados:
                st.session_state["cartelera_activa"] = encontrados
                fechas_enc = sorted({str(x.get("fecha", "")) for x in encontrados})
                if str(fecha_busqueda) in fechas_enc:
                    st.success(f"Se cargaron {len(encontrados)} partidos oficiales desde ESPN NBA ({', '.join(fechas_enc)}).")
                else:
                    st.info(
                        f"En la fecha seleccionada ({fecha_busqueda}) no hubo jornada NBA; "
                        f"se cargaron automáticamente {len(encontrados)} partidos oficiales de la jornada más cercana en ESPN ({', '.join(fechas_enc)})."
                    )
            else:
                st.warning("No se encontraron partidos en vivo en ESPN (temporada en pausa o sin conexión); puedes usar la cartelera cargada o añadir cruces manualmente abajo.")

    # Gestión manual de cruces (añadir o limpiar)
    with st.expander("➕ Añadir o Quitar Partidos Manualmente a la Jornada", expanded=False):
        teams_all = sorted(list(NBA_TEAMS.keys()))
        ca1, ca2, ca3 = st.columns([2, 2, 1])
        with ca1:
            add_loc = st.selectbox("Local a añadir", options=teams_all, index=0, key="add_loc")
        with ca2:
            add_vis = st.selectbox("Visita a añadir", options=teams_all, index=1, key="add_vis")
        with ca3:
            st.write("")
            st.write("")
            if st.button("➕ Añadir Cruce", use_container_width=True):
                if add_loc != add_vis:
                    st.session_state["cartelera_activa"].append(
                        {"fecha": str(fecha_busqueda), "local": add_loc, "visita": add_vis}
                    )
        if st.button("🗑️ Limpiar Cartelera"):
            st.session_state["cartelera_activa"] = []

    cartelera = st.session_state.get("cartelera_activa", [])
    if not cartelera:
        st.info("Agrega al menos 2 partidos a la cartelera o pulsa 'Buscar Partidos en ESPN NBA'.")
        return

    # Analizar todos los partidos de la noche (con caché en session_state para navegación instantánea)
    combo_cache_key = (
        str([(g.get("local"), g.get("visita"), g.get("fecha")) for g in cartelera])
        + f"_{params['casa_apuestas']}_{params['half_life']}_{params['n_sims']}"
    )
    if st.session_state.get("combo_cache_key") != combo_cache_key or "ultimo_res_combos" not in st.session_state:
        predicciones_noche: List[Dict[str, Any]] = []
        for idx_g, g in enumerate(cartelera):
            loc_g = normalizar_equipo(g.get("local"))
            vis_g = normalizar_equipo(g.get("visita"))
            if not loc_g or not vis_g or loc_g == vis_g:
                continue
            p_g = engine.predecir_partido_completo(
                local=loc_g,
                visita=vis_g,
                fecha=g.get("fecha", str(fecha_busqueda)),
                casa_apuestas=params["casa_apuestas"],
                n_sims=params["n_sims"],
                seed=42 + idx_g,
            )
            predicciones_noche.append(p_g)

        st.session_state["ultimo_res_combos"] = armar_combinadas_y_zona_picante(
            predicciones_noche, casa_apuestas=params["casa_apuestas"]
        )
        st.session_state["combo_cache_key"] = combo_cache_key

    res_combos = st.session_state["ultimo_res_combos"]

    # 2. Ranking de Seguridad de la Noche
    st.markdown("#### 🛡️ 2. Ranking de Seguridad de la Noche (Con Detector de Fatiga B2B y Consenso)")
    df_rank_seg = pd.DataFrame(res_combos["ranking_seguridad"])
    if not df_rank_seg.empty:
        df_rank_view = pd.DataFrame(
            {
                "Partido": df_rank_seg["partido"],
                "Selección Más Segura": df_rank_seg["seleccion_principal"],
                "Categoría": df_rank_seg["categoria"],
                "Probabilidad %": (df_rank_seg["probabilidad"] * 100).round(1),
                "Piso P10 %": (df_rank_seg["piso_p10"] * 100).round(1),
                "Cuota Justa": df_rank_seg["cuota_justa"],
                "Fatiga / Altitud": df_rank_seg["alerta_fatiga"],
                "Consenso": df_rank_seg["estado_consenso"],
            }
        )
        st.dataframe(df_rank_view, use_container_width=True, hide_index=True)

    # 3. Mejor Combinada Doble y Mejor Combinada Triple
    st.markdown("#### 🏆 3. Armador Automático de Combinadas Principales (Sin Partidos Trampa)")
    cd1, cd2 = st.columns(2)
    with cd1:
        doble = res_combos["mejor_doble"]
        if doble:
            st.success(f"**{doble['titulo']}**")
            for s in doble["selecciones"]:
                st.write(
                    f"- **{s['partido']}** → `{s['mercado']}` "
                    f"(`{s['probabilidad']*100:.1f}%` | P10 `{s['piso_p10']*100:.1f}%` | Cuota `{s['cuota_individual']:.2f}`)"
                )
            st.markdown(
                f"**Probabilidad Conjunta:** `{doble['probabilidad_conjunta']*100:.1f}%` "
                f"(Piso P10 Conjunto: `{doble['piso_p10_conjunto']*100:.1f}%`)  \n"
                f"**Cuota Justa Combinada:** **`{doble['cuota_combinada']:.2f}`** "
                f"(Paga **`{doble['retorno_por_10u']:.2f} u`** por cada `10 u`)"
            )
        else:
            st.warning("Se necesitan al menos 2 partidos seguros (sin alerta de Partido Trampa) para armar la Doble.")

    with cd2:
        triple = res_combos["mejor_triple"]
        if triple:
            st.info(f"**{triple['titulo']}**")
            for s in triple["selecciones"]:
                st.write(
                    f"- **{s['partido']}** → `{s['mercado']}` "
                    f"(`{s['probabilidad']*100:.1f}%` | P10 `{s['piso_p10']*100:.1f}%` | Cuota `{s['cuota_individual']:.2f}`)"
                )
            st.markdown(
                f"**Probabilidad Conjunta:** `{triple['probabilidad_conjunta']*100:.1f}%` "
                f"(Piso P10 Conjunto: `{triple['piso_p10_conjunto']*100:.1f}%`)  \n"
                f"**Cuota Justa Combinada:** **`{triple['cuota_combinada']:.2f}`** "
                f"(Paga **`{triple['retorno_por_10u']:.2f} u`** por cada `10 u`)"
            )
        else:
            st.warning("Se necesitan al menos 3 partidos seguros (sin alerta de Partido Trampa) para armar la Triple.")

    # 4. 🌶️ Zona Picante: El Boleto de Cuota 5.00+
    st.markdown("---")
    st.markdown("### 🌶️ Zona Picante: El Boleto de Cuota 5.00+ (Juego Recreativo / Cash Rápido)")
    st.caption(
        "⚠️ Advertencia Cuantitativa: Los boletos de cuota >= 5.00 tienen mayor varianza y están "
        "diseñados exclusivamente para uso recreativo con stakes bajos (0.25u - 0.50u)."
    )

    zp = res_combos["zona_picante"]
    cz1, cz2 = st.columns(2)
    with cz1:
        esc = zp.get("escalera_picante")
        if esc:
            st.warning(f"**{esc['titulo']}**")
            for s in esc["selecciones"]:
                st.write(
                    f"- **{s['partido']}** → `{s['mercado']}` "
                    f"(Prob: `{s['probabilidad']*100:.1f}%` | Cuota: `{s['cuota_individual']:.2f}`)"
                )
            st.markdown(
                f"🔥 **Cuota Combinada Total:** **`{esc['cuota_combinada']:.2f}`**  \n"
                f"📊 **Probabilidad Conjunta:** `{esc['probabilidad_conjunta']*100:.1f}%`  \n"
                f"💵 **Retorno por cada 10 unidades apostadas:** **`{esc['retorno_por_10u']:.2f} u`**"
            )
    with cz2:
        bomba = zp.get("bomba_express")
        if bomba:
            st.error(f"**{bomba['titulo']}**")
            for s in bomba["selecciones"]:
                st.write(
                    f"- **{s['partido']}** → `{s['mercado']}` "
                    f"(Prob: `{s['probabilidad']*100:.1f}%` | Cuota: `{s['cuota_individual']:.2f}`)"
                )
            st.markdown(
                f"💣 **Cuota Combinada Total:** **`{bomba['cuota_combinada']:.2f}`**  \n"
                f"📊 **Probabilidad Conjunta:** `{bomba['probabilidad_conjunta']*100:.1f}%`  \n"
                f"💵 **Retorno por cada 10 unidades apostadas:** **`{bomba['retorno_por_10u']:.2f} u`**"
            )


def render_tab_power_ranking(engine: NBAEnsembleEngine) -> None:
    """Pestaña 4: 🏅 Ranking NBA (Power Ranking ordenado por Elo / Net Rating y Top 15 en gráfico)."""
    st.subheader("🏅 Power Ranking NBA (Elo FiveThirtyEight & Eficiencia EWMA)")
    df_rank = generar_tabla_power_ranking(engine.tracker, engine.elo_sys)

    if df_rank.empty:
        st.info("No hay datos suficientes para generar el Power Ranking.")
        return

    st.dataframe(df_rank, use_container_width=True, hide_index=True)

    st.markdown("#### 📊 Top 15 Franquicias NBA por Rating Elo")
    top15 = df_rank.head(15).set_index("Equipo")[["Elo", "Net Rating"]]
    st.bar_chart(top15[["Elo"]])


def render_tab_backtest(df_partidos: pd.DataFrame, params: Dict[str, Any]) -> None:
    """Pestaña 5: 📈 Backtest Walk-Forward Cronológico y Simulación Financiera de Banca."""
    st.subheader("📈 Evaluación Walk-Forward Estrictamente Cronológica (Zero Data Leakage)")
    st.caption(
        "Compara el Ensemble de 3 Pilares contra Solo Elo y Línea Base en Log Loss, Brier Score, "
        "Accuracy %, MAE del Spread, Curva de Calibración y Simulación Financiera de Banca."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        warmup = st.slider("Partidos de Calentamiento (Warmup)", min_value=20, max_value=150, value=60, step=10)
    with c2:
        umbral_ev = st.slider("Umbral Mínimo EV % para Apostar", min_value=0.5, max_value=8.0, value=2.0, step=0.5)
    with c3:
        st.write("")
        st.write("")
        ejecutar_btn = st.button("🚀 Ejecutar Backtest Walk-Forward", use_container_width=True)

    if ejecutar_btn or "ultimo_backtest" not in st.session_state:
        max_f = str(df_partidos["fecha"].max()) if not df_partidos.empty else ""
        if hasattr(st, "cache_data"):
            st.session_state["ultimo_backtest"] = _ejecutar_backtest_cached(
                n_rows=len(df_partidos),
                max_fecha=max_f,
                warmup_games=warmup,
                half_life=params["half_life"],
                umbral_ev_pct=umbral_ev,
                _df_partidos=df_partidos,
            )
        else:
            st.session_state["ultimo_backtest"] = ejecutar_backtest_walk_forward(
                df_partidos=df_partidos,
                warmup_games=warmup,
                half_life=params["half_life"],
                umbral_ev_pct=umbral_ev,
            )

    bt = st.session_state["ultimo_backtest"]
    if bt.get("status") != "ok":
        st.warning(bt.get("mensaje", "No se pudo ejecutar el backtest."))
        return

    st.markdown(f"#### 📋 Comparativa de Modelos (`{bt['partidos_evaluados']}` partidos evaluados fuera de muestra)")
    st.dataframe(bt["tabla_comparativa"], use_container_width=True, hide_index=True)

    sim = bt["simulacion_financiera"]
    st.markdown("#### 💰 Simulación Financiera de Banca (Estrategia Cuantitativa EV+)")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Apuestas Realizadas", f"{sim['apuestas_realizadas']}")
    s2.metric("Win Rate %", f"{sim['win_rate_pct']:.1f}%")
    s3.metric("Unidades Ganadas", f"{sim['unidades_ganadas']:+.2f} u")
    s4.metric("ROI / Yield %", f"{sim['roi_yield_pct']:+.2f}%")
    s5.metric("Max Drawdown", f"-{sim['max_drawdown_u']:.2f} u")

    col_cal, col_evol = st.columns([1.2, 1.8])
    with col_cal:
        st.markdown("##### 🎯 Curva de Calibración Probabilística")
        st.dataframe(bt["curva_calibracion"], use_container_width=True, hide_index=True)
    with col_evol:
        st.markdown("##### 📈 Evolución Acumulada de Ganancias (Unidades)")
        df_hist_ap = sim["historial_apuestas"]
        if not df_hist_ap.empty:
            st.line_chart(df_hist_ap.set_index("Apuesta #")[["Unidades Acumuladas"]])
        else:
            st.info("Ningún partido superó el umbral de EV% seleccionado.")


def render_tab_jugadores(
    engine: NBAEnsembleEngine, params: Dict[str, Any]
) -> None:
    """
    Pestaña dedicada: ⛹️‍♂️ Jugadores & Props
    - Proyecciones y Simulación de Monte Carlo para Puntos (PTS), Rebotes (REB),
      Asistencias (AST), Triples (3PM) y Combos PRA (PTS+REB+AST).
    - Calculadora individual de líneas y cuotas (Over/Under + EV% + 1/4 Kelly) ideal para
      Betano y Betsafe Bet Builder.
    - Tabla de líderes y catálogo de estrellas NBA.
    """
    st.subheader("⛹️‍♂️ Estadísticas de Jugadores & Laboratorio de Player Props (PTS, REB, AST, 3PM, PRA)")

    pred = st.session_state.get("ultima_prediccion")
    if not pred:
        pred = engine.predecir_partido_completo(
            local="Boston Celtics",
            visita="Los Angeles Lakers",
            casa_apuestas=params["casa_apuestas"],
            n_sims=params["n_sims"],
        )

    local = pred["local"]
    visita = pred["visita"]
    pilares = pred["pilares"]
    ft = pred["features"]
    bajas_l = st.session_state.get("ultimas_bajas_l", [])
    bajas_v = st.session_state.get("ultimas_bajas_v", [])

    col_cap, col_sync = st.columns([3, 1.3])
    with col_cap:
        st.caption(
            f"Contexto activo del partido: **{visita} @ {local}** | "
            f"Ritmo Esperado (Pace): **{pilares['pace_esperado']:.1f} poss** | "
            f"DRtg {local}: **{ft['drtg_l']:.1f}** · DRtg {visita}: **{ft['drtg_v']:.1f}**"
        )
    with col_sync:
        if st.button("🔄 Sincronizar Jugadores y Bajas (ESPN Live)", use_container_width=True, key="btn_sync_jug_tab"):
            barra_j = st.progress(0.0, text="Conectando con ESPN NBA (Plantillas, Lesiones y Boxscores L5)...")
            res_sync = sincronizar_jugadores_espn(
                dias_boxscores=14,
                max_workers=10,
                progress_callback=lambda p, txt: barra_j.progress(min(1.0, float(p)), text=txt),
            )
            st.success(
                f"¡Sincronizados {res_sync['jugadores_actualizados']} jugadores y "
                f"{res_sync['lesionados_detectados']} reportes médicos desde ESPN Live!"
            )
            st.rerun()

    # 1. Proyección completa de plantillas del partido activo
    plantilla = proyectar_plantilla_partido(
        equipo_local=local,
        equipo_visita=visita,
        pace_esperado=pilares["pace_esperado"],
        drtg_local=ft["drtg_l"],
        drtg_visita=ft["drtg_v"],
        b2b_local=bool(ft["b2b_l"]),
        b2b_visita=bool(ft["b2b_v"]),
        bajas_local=bajas_l,
        bajas_visita=bajas_v,
        n_sims=4000,
    )

    st.markdown("#### 🎯 1. Top Selecciones de Player Props del Partido (Mayor Piso P10 en Monte Carlo)")
    st.dataframe(pd.DataFrame(plantilla["top_props"]), use_container_width=True, hide_index=True)

    # 2. Tablas comparativas de ambos equipos (PTS, REB, AST, 3PM, PRA proyectados)
    col_pl, col_pv = st.columns(2)
    for col_ui, eq_name, lista_jug in [
        (col_pl, f"🏠 {local}", plantilla["jugadores_local"]),
        (col_pv, f"✈️ {visita}", plantilla["jugadores_visita"]),
    ]:
        with col_ui:
            st.markdown(f"##### Plantilla Estelar: {eq_name}")
            filas_eq = []
            for jp in lista_jug:
                mk = jp["mercados"]
                filas_eq.append(
                    {
                        "Jugador": jp["jugador"],
                        "Pos": jp["pos"],
                        "MIN Esp": jp["min_esp"],
                        "PTS Proy": mk["PTS"]["proyeccion"],
                        "REB Proy": mk["REB"]["proyeccion"],
                        "AST Proy": mk["AST"]["proyeccion"],
                        "3PM Proy": mk["3PM"]["proyeccion"],
                        "PRA Proy": mk["PRA"]["proyeccion"],
                        "Mejor Prop": f"{mk['PRA']['recomendacion']} PRA ({mk['PRA']['prob_recomendacion']*100:.0f}%)",
                    }
                )
            st.dataframe(pd.DataFrame(filas_eq), use_container_width=True, hide_index=True)

    # 3. Simulador Individual de Player Props + Calculadora EV+ (Betano / Betsafe)
    st.markdown("---")
    st.markdown("#### 🔬 2. Analizador Individual de Jugador & Calculadora EV+ (PTS, REB, AST, 3PM, PRA)")

    todos_equipos = sorted(list(NBA_TEAMS.keys()))
    c_eq, c_jug, c_opp = st.columns([1.4, 1.6, 1.4])
    with c_eq:
        idx_eq_def = todos_equipos.index(local) if local in todos_equipos else 0
        eq_sel = st.selectbox("Equipo del Jugador", options=todos_equipos, index=idx_eq_def, key="prop_eq_sel")
    roster_sel = obtener_jugadores_equipo(eq_sel)
    nombres_sel = [p["jugador"] for p in roster_sel]
    with c_jug:
        jug_nombre = st.selectbox("Seleccionar Jugador", options=nombres_sel, index=0, key="prop_jug_sel")
    with c_opp:
        idx_opp_def = todos_equipos.index(visita) if (visita in todos_equipos and visita != eq_sel) else 1
        opp_sel = st.selectbox("Rival Enfrentado", options=todos_equipos, index=idx_opp_def, key="prop_opp_sel")

    jug_obj = next((p for p in roster_sel if p["jugador"] == jug_nombre), roster_sel[0])
    perf_eq = engine.tracker.obtener_perfil_hibrido(eq_sel, es_local=True)
    perf_opp = engine.tracker.obtener_perfil_hibrido(opp_sel, es_local=False)
    pace_cruce = 0.42 * perf_eq["pace"] + 0.42 * perf_opp["pace"] + 0.16 * 99.2

    sim_ind = proyectar_y_simular_jugador(
        jugador_info=jug_obj,
        es_local=(eq_sel == local),
        pace_esperado=pace_cruce,
        drtg_rival=perf_opp["drtg"],
        en_b2b=False,
        n_sims=5000,
    )

    st.dataframe(pd.DataFrame(sim_ind["tabla_resumen"]), use_container_width=True, hide_index=True)

    # Calculadora rápida EV+ para una línea específica del jugador seleccionado
    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        mkt_prop_code = st.selectbox(
            "Mercado del Jugador",
            options=["PTS", "REB", "AST", "3PM", "PRA"],
            format_func=lambda k: sim_ind["mercados"][k]["etiqueta"],
            key="prop_mkt_code",
        )
    mkt_actual = sim_ind["mercados"][mkt_prop_code]
    with cp2:
        linea_prop_usr = st.number_input(
            f"Línea de Mercado ({mkt_prop_code})",
            min_value=0.5,
            max_value=80.5,
            value=float(mkt_actual["linea"]),
            step=1.0,
            key="prop_linea_usr",
        )
    with cp3:
        cuota_prop_usr = st.number_input(
            "Cuota Casa (Betano / Betsafe)",
            min_value=1.05,
            max_value=15.0,
            value=1.87,
            step=0.02,
            key="prop_cuota_usr",
        )

    sim_custom = proyectar_y_simular_jugador(
        jugador_info=jug_obj,
        es_local=(eq_sel == local),
        pace_esperado=pace_cruce,
        drtg_rival=perf_opp["drtg"],
        lineas_custom={mkt_prop_code: float(linea_prop_usr)},
        n_sims=5000,
    )
    m_res = sim_custom["mercados"][mkt_prop_code]
    ev_over_p = calcular_ev_y_kelly(m_res["prob_over"], cuota_prop_usr, prob_p25=m_res["p10_over"])
    ev_under_p = calcular_ev_y_kelly(m_res["prob_under"], cuota_prop_usr, prob_p25=m_res["p10_under"])

    co1, co2 = st.columns(2)
    with co1:
        st.info(
            f"📈 **OVER (Más de {linea_prop_usr:.1f} {mkt_prop_code}) — {jug_nombre}**\n"
            f"- Probabilidad Monte Carlo: **`{m_res['prob_over']*100:.1f}%`** (Piso P10: `{m_res['p10_over']*100:.1f}%`)\n"
            f"- Cuota Justa: **`{m_res['cuota_justa_over']:.2f}`** | EV: **`{ev_over_p['ev_pct']:+.2f}%`**\n"
            f"- Stake 1/4 Kelly: `{ev_over_p['stake_kelly_pct']:.2f}%` ({ev_over_p['recomendacion']})"
        )
    with co2:
        st.info(
            f"📉 **UNDER (Menos de {linea_prop_usr:.1f} {mkt_prop_code}) — {jug_nombre}**\n"
            f"- Probabilidad Monte Carlo: **`{m_res['prob_under']*100:.1f}%`** (Piso P10: `{m_res['p10_under']*100:.1f}%`)\n"
            f"- Cuota Justa: **`{m_res['cuota_justa_under']:.2f}`** | EV: **`{ev_under_p['ev_pct']:+.2f}%`**\n"
            f"- Stake 1/4 Kelly: `{ev_under_p['stake_kelly_pct']:.2f}%` ({ev_under_p['recomendacion']})"
        )

    # 4. Catálogo Completo de Líderes NBA (PTS, REB, AST, 3PM, PRA)
    with st.expander("🏆 Ver Catálogo Completo y Ranking de Jugadores NBA (30 Franquicias)", expanded=False):
        df_cat_jug = obtener_catalogo_completo_jugadores()
        st.dataframe(df_cat_jug, use_container_width=True, hide_index=True)


def render_tab_datos(df_partidos: pd.DataFrame) -> None:
    """Pestaña 6: 🗄️ Datos (Ingesta Multihilo ESPN, Actualización Diaria e Importador CSV/Excel)."""
    st.subheader("🗄️ Gestión de Base de Datos SQLite & Ingesta ESPN NBA (Sin API Keys)")

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Partidos en SQLite", f"{len(df_partidos):,}")
    temporadas_unicas = sorted(df_partidos["temporada"].dropna().unique().tolist()) if not df_partidos.empty else []
    d2.metric("Temporadas Cargadas", ", ".join(str(int(t)) for t in temporadas_unicas) or "Ninguna")
    ult_fecha = str(df_partidos["fecha"].max()) if not df_partidos.empty else "—"
    d3.metric("Último Partido Registrado", ult_fecha)

    col_ing1, col_ing2 = st.columns(2)
    with col_ing1:
        st.markdown("#### 🌐 1. Descarga Histórica Multihilo (ESPN NBA)")
        temp_sel = st.multiselect(
            "Selecciona Temporadas a Descargar (30 Franquicias en Paralelo)",
            options=[2023, 2024, 2025, 2026],
            default=[2025, 2026],
        )
        if st.button("⬇️ Descargar Historial Completo desde ESPN", use_container_width=True):
            barra = st.progress(0.0, text="Iniciando descarga paralela desde ESPN...")
            n_guardados = descargar_historial_nba(
                temporadas=temp_sel,
                max_workers=10,
                progress_callback=lambda p, txt: barra.progress(min(1.0, float(p)), text=txt),
            )
            if hasattr(st, "cache_resource"):
                st.cache_resource.clear()
            if hasattr(st, "cache_data"):
                st.cache_data.clear()
            st.success(f"¡Descarga completada! Se procesaron y almacenaron {n_guardados} partidos en SQLite.")
            st.rerun()

        if st.button("🔄 Actualizar Últimos 7 Días (Scoreboard & Cuartos Q1/1H)", use_container_width=True):
            n_rec = actualizar_partidos_recientes(dias_atras=7)
            if hasattr(st, "cache_resource"):
                st.cache_resource.clear()
            if hasattr(st, "cache_data"):
                st.cache_data.clear()
            st.success(f"Se actualizaron {n_rec} partidos recientes con detalle de cuartos y boxscore.")
            st.rerun()

        if st.button("⛹️‍♂️ Sincronizar Plantillas, Lesiones y Racha L5 (ESPN Live)", use_container_width=True):
            barra_p = st.progress(0.0, text="Descargando plantillas, lesionados y boxscores de jugadores desde ESPN...")
            res_j = sincronizar_jugadores_espn(
                dias_boxscores=14,
                max_workers=10,
                progress_callback=lambda p, txt: barra_p.progress(min(1.0, float(p)), text=txt),
            )
            st.success(
                f"Sincronización completada: {res_j['jugadores_actualizados']} jugadores actualizados, "
                f"{res_j['lesionados_detectados']} reportes de lesión y {res_j['partidos_boxscore_analizados']} boxscores procesados."
            )
            st.rerun()

    with col_ing2:
        st.markdown("#### 📂 2. Importar Archivo Propio (`.csv` o `.xlsx`)")
        archivo_subido = st.file_uploader(
            "Sube tu archivo CSV o Excel con historial o cuotas propias",
            type=["csv", "xlsx", "xls"],
        )
        if archivo_subido is not None:
            if st.button("📥 Importar Archivo a SQLite", use_container_width=True):
                try:
                    res_imp = importar_archivo_usuario(archivo_subido, nombre_archivo=archivo_subido.name)
                    if hasattr(st, "cache_resource"):
                        st.cache_resource.clear()
                    if hasattr(st, "cache_data"):
                        st.cache_data.clear()
                    st.success(res_imp["mensaje"])
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error al importar archivo: {exc}")

    st.markdown("#### 🔍 Vista Previa de los Últimos Partidos Almacenados (Con Boxscore y Ratings)")
    if not df_partidos.empty:
        st.dataframe(df_partidos.tail(25).iloc[::-1], use_container_width=True, hide_index=True)


def main() -> None:
    configurar_pagina()
    params = render_sidebar()

    engine, df_partidos, _ = cargar_motor_entrenado(
        half_life=params["half_life"],
        n_sims=params["n_sims"],
    )

    st.title("🏀 NBA ULTIMATE AI")
    st.caption(
        "Sistema Cuantitativo de Predicción Probabilística, Simulación de Monte Carlo Correlacionada "
        "y Detección de Valor (EV+) optimizado para **Betano** y **Betsafe**."
    )

    if st.session_state.get("foco_combinadas"):
        st.info("🎯 Has activado el acceso rápido a **Combinadas**. Selecciona la pestaña **🎯 Combinadas** abajo o explora la cartelera.")

    tabs = st.tabs(
        [
            "🔮 Predicción",
            "⛹️‍♂️ Jugadores & Props",
            "💹 EV+ & Kelly",
            "🎯 Combinadas",
            "🏅 Ranking NBA",
            "📈 Backtest",
            "🗄️ Datos",
        ]
    )

    with tabs[0]:
        render_tab_prediccion(engine, params)
    with tabs[1]:
        render_tab_jugadores(engine, params)
    with tabs[2]:
        render_tab_ev_kelly(engine, params)
    with tabs[3]:
        render_tab_combinadas(engine, params)
    with tabs[4]:
        render_tab_power_ranking(engine)
    with tabs[5]:
        render_tab_backtest(df_partidos, params)
    with tabs[6]:
        render_tab_datos(df_partidos)


if __name__ == "__main__":
    main()
