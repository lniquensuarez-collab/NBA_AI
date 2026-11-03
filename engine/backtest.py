"""
Módulo de Backtest Walk-Forward Estrictamente Cronológico (`engine/backtest.py`):
- Evaluación walk-forward sin fuga de información (Zero Data Leakage) comparando:
    1. Ensemble NBA Ultimate (3 Pilares: Posesional/Eficiencia + Elo MOV + Four Factors EWMA)
    2. Solo Elo NBA (FiveThirtyEight MOV)
    3. Línea Base (Ventaja histórica de localía constante)
- Métricas probabilísticas y de regresión:
    * Log Loss
    * Brier Score
    * Accuracy % (Acierto de ganador)
    * MAE del Spread (Error medio absoluto en puntos)
    * MAE de Totales (Error medio absoluto del total de puntos)
- Curva de Calibración por deciles/bins de probabilidad.
- Simulación Financiera de Banca (ROI / Yield %, Unidades Ganadas, Win Rate %, Max Drawdown
  y serie temporal de evolución de banca).
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from engine.config import BASE_STD_DEV_DIFF, DEFAULT_HALF_LIFE
from engine.features import construir_dataset_cronologico, FEATURE_COLS_ML


def _calcular_metricas_modelo(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    margin_true: np.ndarray,
    spread_pred: np.ndarray,
) -> Dict[str, float]:
    p_safe = np.clip(p_pred, 1e-4, 1.0 - 1e-4)
    log_loss = float(-np.mean(y_true * np.log(p_safe) + (1.0 - y_true) * np.log(1.0 - p_safe)))
    brier = float(np.mean((p_safe - y_true) ** 2))
    acc = float(np.mean((p_safe >= 0.5) == (y_true == 1)) * 100.0)
    mae_spread = float(np.mean(np.abs(margin_true - spread_pred)))
    return {
        "log_loss": round(log_loss, 4),
        "brier_score": round(brier, 4),
        "accuracy_pct": round(acc, 2),
        "mae_spread_pts": round(mae_spread, 2),
    }


def ejecutar_backtest_walk_forward(
    df_partidos: pd.DataFrame,
    warmup_games: int = 60,
    retrain_step: int = 25,
    half_life: float = DEFAULT_HALF_LIFE,
    umbral_ev_pct: float = 2.0,
) -> Dict[str, Any]:
    """
    Ejecuta un backtest walk-forward estrictamente cronológico:
    - Los primeros `warmup_games` partidos sirven como ventana inicial de calentamiento.
    - Para cada partido `i >= warmup_games`, el modelo se entrena únicamente con los partidos
      `[0 .. i-1]` (re-entrenando cada `retrain_step` partidos para velocidad óptima)
      y predice el partido `i` sin ver jamás el futuro.
    """
    if df_partidos.empty or len(df_partidos) < 25:
        return {
            "status": "insuficiente",
            "mensaje": "Se requieren al menos 25 partidos históricos para ejecutar el Backtest Walk-Forward.",
        }

    df_feat, _, _ = construir_dataset_cronologico(df_partidos)
    n_total = len(df_feat)
    warmup = min(max(20, int(warmup_games)), max(20, int(n_total * 0.35)))

    scaler = StandardScaler()
    clf = LogisticRegression(C=0.45, max_iter=400)
    reg = Ridge(alpha=18.0)
    is_trained = False

    registros_eval: List[Dict[str, Any]] = []
    # Tasa histórica de victoria local acumulada para la Línea Base
    wins_local_acum = int(df_feat.iloc[:warmup]["target_win_l"].sum())
    margin_local_acum = float(df_feat.iloc[:warmup]["target_margin_l"].mean())

    for i in range(warmup, n_total):
        # Re-entrenar modelos supervisados con datos estrictamente anteriores [0 : i]
        if (not is_trained) or ((i - warmup) % max(5, int(retrain_step)) == 0):
            train_slice = df_feat.iloc[:i]
            X_tr = train_slice[FEATURE_COLS_ML].fillna(0.0).to_numpy(dtype=float)
            y_tr_win = train_slice["target_win_l"].to_numpy(dtype=int)
            y_tr_mar = train_slice["target_margin_l"].to_numpy(dtype=float)

            fechas_tr = pd.to_datetime(train_slice["fecha"])
            dias_tr = (fechas_tr.max() - fechas_tr).dt.days.clip(lower=0).to_numpy(dtype=float)
            w_tr = np.exp(-(np.log(2.0) / max(15.0, float(half_life))) * dias_tr)
            w_tr = np.clip(w_tr, 0.05, 1.0)

            if len(np.unique(y_tr_win)) >= 2:
                X_tr_sc = scaler.fit_transform(X_tr)
                clf.fit(X_tr_sc, y_tr_win, sample_weight=w_tr)
                reg.fit(X_tr_sc, y_tr_mar, sample_weight=w_tr)
                is_trained = True

            wins_local_acum = int(train_slice["target_win_l"].sum())
            margin_local_acum = float(train_slice["target_margin_l"].mean())

        row = df_feat.iloc[i]

        # 1. Predicción Línea Base (frecuencia histórica previa de victoria local)
        p_base = float(np.clip(wins_local_acum / max(1, i), 0.45, 0.65))
        spread_base = float(margin_local_acum)

        # 2. Predicción Solo Elo (calculada estrictamente antes del partido i)
        p_elo = float(np.clip(row["p_elo"], 0.01, 0.99))
        spread_elo = float(row["spread_elo"])

        # 3. Predicción Pilar Posesional / Eficiencia EWMA pre-partido
        pace_esp = 0.42 * float(row["pace_l"]) + 0.42 * float(row["pace_v"]) + 0.16 * 99.2
        eff_diff = (
            0.5 * ((float(row["ortg_l"]) - float(row["drtg_v"])) - (float(row["ortg_v"]) - float(row["drtg_l"])))
            + 2.4
            + float(row["impacto_tactico_pts"])
        )
        spread_pos = (pace_esp / 100.0) * eff_diff
        p_pos = float(norm.cdf(spread_pos / BASE_STD_DEV_DIFF))

        # 4. Predicción Pilar ML Four Factors
        if is_trained:
            x_now = np.array([[float(row[c]) for c in FEATURE_COLS_ML]], dtype=float)
            x_now_sc = scaler.transform(x_now)
            p_ml = float(clf.predict_proba(x_now_sc)[0, 1])
            spread_ml = float(reg.predict(x_now_sc)[0])
        else:
            spread_ml = spread_pos
            p_ml = p_pos

        # Combinación Ensemble (45% Posesional, 30% ML Four Factors, 25% Elo MOV)
        spread_ens = 0.45 * spread_pos + 0.30 * spread_ml + 0.25 * spread_elo
        p_ens_raw = 0.45 * p_pos + 0.30 * p_ml + 0.25 * p_elo
        p_ens_norm = float(norm.cdf(spread_ens / BASE_STD_DEV_DIFF))
        p_ens = float(np.clip(0.60 * p_ens_raw + 0.40 * p_ens_norm, 0.01, 0.99))

        total_pred = (pace_esp / 100.0) * (
            0.5 * (float(row["ortg_l"]) + float(row["drtg_v"]))
            + 0.5 * (float(row["ortg_v"]) + float(row["drtg_l"]))
        )

        # Cuotas de mercado (reales del dataset o sintéticas realistas basadas en consenso Elo/Base con vig 4.5%)
        cuota_l = row.get("cuota_l")
        cuota_v = row.get("cuota_v")
        if cuota_l is None or pd.isna(cuota_l) or float(cuota_l or 0) <= 1.01:
            p_mkt_l = float(np.clip(0.75 * p_elo + 0.25 * p_base, 0.08, 0.92))
            cuota_l = round(max(1.04, 1.0 / (p_mkt_l * 1.045)), 2)
            cuota_v = round(max(1.04, 1.0 / ((1.0 - p_mkt_l) * 1.045)), 2)
        else:
            cuota_l = float(cuota_l)
            cuota_v = float(cuota_v) if (cuota_v is not None and pd.notna(cuota_v)) else round(1.0 / max(0.05, 1.045 - (1.0 / cuota_l)), 2)

        registros_eval.append(
            {
                "fecha": row["fecha"],
                "local": row["local"],
                "visita": row["visita"],
                "pts_l": int(row["pts_l"]),
                "pts_v": int(row["pts_v"]),
                "y_win_l": int(row["target_win_l"]),
                "margin_real": float(row["target_margin_l"]),
                "total_real": float(row["target_total"]),
                "p_ens": p_ens,
                "spread_ens": spread_ens,
                "total_ens": total_pred,
                "p_elo": p_elo,
                "spread_elo": spread_elo,
                "p_base": p_base,
                "spread_base": spread_base,
                "cuota_l": cuota_l,
                "cuota_v": cuota_v,
            }
        )

    df_eval = pd.DataFrame(registros_eval)
    y_true = df_eval["y_win_l"].to_numpy(dtype=float)
    margin_true = df_eval["margin_real"].to_numpy(dtype=float)

    m_ens = _calcular_metricas_modelo(
        y_true, df_eval["p_ens"].to_numpy(), margin_true, df_eval["spread_ens"].to_numpy()
    )
    m_elo = _calcular_metricas_modelo(
        y_true, df_eval["p_elo"].to_numpy(), margin_true, df_eval["spread_elo"].to_numpy()
    )
    m_base = _calcular_metricas_modelo(
        y_true, df_eval["p_base"].to_numpy(), margin_true, df_eval["spread_base"].to_numpy()
    )
    mae_total_ens = float(np.mean(np.abs(df_eval["total_real"] - df_eval["total_ens"])))
    m_ens["mae_total_pts"] = round(mae_total_ens, 2)

    tabla_comparativa = pd.DataFrame(
        [
            {
                "Modelo": "🏀 Ensemble NBA Ultimate (3 Pilares)",
                "Log Loss": m_ens["log_loss"],
                "Brier Score": m_ens["brier_score"],
                "Accuracy %": m_ens["accuracy_pct"],
                "MAE Spread (pts)": m_ens["mae_spread_pts"],
            },
            {
                "Modelo": "⚡ Solo Sistema Elo (MOV)",
                "Log Loss": m_elo["log_loss"],
                "Brier Score": m_elo["brier_score"],
                "Accuracy %": m_elo["accuracy_pct"],
                "MAE Spread (pts)": m_elo["mae_spread_pts"],
            },
            {
                "Modelo": "📏 Línea Base (Localía Constante)",
                "Log Loss": m_base["log_loss"],
                "Brier Score": m_base["brier_score"],
                "Accuracy %": m_base["accuracy_pct"],
                "MAE Spread (pts)": m_base["mae_spread_pts"],
            },
        ]
    )

    # Curva de Calibración (5 bins)
    bins = np.linspace(0.0, 1.0, 6)
    df_eval["bin"] = pd.cut(df_eval["p_ens"], bins=bins, include_lowest=True)
    cal_rows = []
    for b_label, grp in df_eval.groupby("bin", observed=False):
        if len(grp) > 0:
            cal_rows.append(
                {
                    "Rango Probabilidad": str(b_label),
                    "Partidos": int(len(grp)),
                    "Prob. Predicha Media": round(float(grp["p_ens"].mean()) * 100.0, 1),
                    "Frecuencia Real Victoria %": round(float(grp["y_win_l"].mean()) * 100.0, 1),
                }
            )
    df_calibracion = pd.DataFrame(cal_rows)

    # Simulación Financiera de Banca (Estrategia Cuantitativa EV+ / Consenso)
    banca_evolucion = []
    unidades_acum = 0.0
    pico_unidades = 0.0
    max_drawdown = 0.0
    apuestas_totales = 0
    apuestas_ganadas = 0
    total_apostado_u = 0.0

    for _, r in df_eval.iterrows():
        p_l = float(r["p_ens"])
        p_v = 1.0 - p_l
        ev_l = (p_l * float(r["cuota_l"]) - 1.0) * 100.0
        ev_v = (p_v * float(r["cuota_v"]) - 1.0) * 100.0

        apostar_lado = None
        cuota_sel = 1.0
        prob_sel = 0.5
        gano_apuesta = False

        if ev_l >= umbral_ev_pct and p_l >= 0.52:
            apostar_lado = "Local"
            cuota_sel = float(r["cuota_l"])
            prob_sel = p_l
            gano_apuesta = int(r["y_win_l"]) == 1
        elif ev_v >= umbral_ev_pct and p_v >= 0.52:
            apostar_lado = "Visita"
            cuota_sel = float(r["cuota_v"])
            prob_sel = p_v
            gano_apuesta = int(r["y_win_l"]) == 0

        if apostar_lado is not None:
            # Stake plano de 1.0 unidad para medición transparente de Yield/ROI
            stake_u = 1.0
            apuestas_totales += 1
            total_apostado_u += stake_u
            if gano_apuesta:
                apuestas_ganadas += 1
                pnl = stake_u * (cuota_sel - 1.0)
            else:
                pnl = -stake_u

            unidades_acum += pnl
            pico_unidades = max(pico_unidades, unidades_acum)
            dd_actual = pico_unidades - unidades_acum
            max_drawdown = max(max_drawdown, dd_actual)

            banca_evolucion.append(
                {
                    "Apuesta #": apuestas_totales,
                    "Fecha": r["fecha"],
                    "Partido": f"{r['visita']} @ {r['local']}",
                    "Selección": apostar_lado,
                    "Cuota": round(cuota_sel, 2),
                    "Prob. Modelo %": round(prob_sel * 100.0, 1),
                    "Resultado": "✅ Ganada" if gano_apuesta else "❌ Perdida",
                    "PnL (u)": round(pnl, 2),
                    "Unidades Acumuladas": round(unidades_acum, 2),
                }
            )

    win_rate_pct = (apuestas_ganadas / max(1, apuestas_totales)) * 100.0 if apuestas_totales > 0 else 0.0
    roi_yield_pct = (unidades_acum / max(1.0, total_apostado_u)) * 100.0 if apuestas_totales > 0 else 0.0

    return {
        "status": "ok",
        "partidos_evaluados": len(df_eval),
        "warmup_games": warmup,
        "metricas_ensemble": m_ens,
        "metricas_elo": m_elo,
        "metricas_base": m_base,
        "tabla_comparativa": tabla_comparativa,
        "curva_calibracion": df_calibracion,
        "simulacion_financiera": {
            "apuestas_realizadas": apuestas_totales,
            "apuestas_ganadas": apuestas_ganadas,
            "win_rate_pct": round(win_rate_pct, 2),
            "unidades_ganadas": round(unidades_acum, 2),
            "roi_yield_pct": round(roi_yield_pct, 2),
            "max_drawdown_u": round(max_drawdown, 2),
            "historial_apuestas": pd.DataFrame(banca_evolucion),
        },
        "df_evaluacion": df_eval,
    }
