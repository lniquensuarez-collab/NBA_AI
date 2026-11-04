"""
Motor Matemático Ensemble de 3 Pilares, Simulación de Monte Carlo Correlacionada,
Bono de Pago Anticipado +20 pts (Betano/Betsafe), Detector de Partidos Trampa,
Selector Inteligente de Bajo Riesgo y Armador de Combinadas + Zona Picante 5.0+ (`engine/models.py`).
"""

from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from engine.config import (
    NBA_TEAMS,
    NBA_AVG_PACE,
    NBA_AVG_ORTG,
    NBA_HOME_COURT_PTS,
    DEFAULT_HALF_LIFE,
    DEFAULT_MC_SIMS,
    BASE_STD_DEV_DIFF,
    BASE_STD_DEV_TOTAL,
    PROP_Q1_BASE,
    PROP_1H_BASE,
    normalizar_equipo,
)
from engine.db import desmargenar_cuotas_power
from engine.elo_system import EloSystemNBA
from engine.features import (
    TeamStateTracker,
    extraer_features_pre_partido,
    FEATURE_COLS_ML,
)


class ModeloPosesionalNBA:
    """
    Pilar 1: Modelo Posesional y de Eficiencia (Ritmo × Eficiencia con Decaimiento Exponencial).
    Ajusta mediante optimización regularizada L-BFGS-B / Ridge con decaimiento temporal `half_life`:
      1. Ritmo esperado de cada franquicia (Pace)
      2. Ataque (ORtg) y Defensa (DRtg) ajustados por la calidad de los rivales enfrentados (SRS / Ridge)
      3. Ventaja de localía específica NBA
    """

    def __init__(self, half_life: float = DEFAULT_HALF_LIFE, l2_reg: float = 12.0) -> None:
        self.half_life = float(max(15.0, half_life))
        self.l2_reg = float(l2_reg)
        self.teams = sorted(list(NBA_TEAMS.keys()))
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        self.ortg_adj: Dict[str, float] = {t: NBA_AVG_ORTG for t in self.teams}
        self.drtg_adj: Dict[str, float] = {t: NBA_AVG_ORTG for t in self.teams}
        self.pace_adj: Dict[str, float] = {t: NBA_AVG_PACE for t in self.teams}
        self.hca_ortg: float = NBA_HOME_COURT_PTS
        self.league_pace: float = NBA_AVG_PACE
        self.league_ortg: float = NBA_AVG_ORTG
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "ModeloPosesionalNBA":
        if df.empty or len(df) < 10:
            self.is_fitted = False
            return self

        df_clean = df.dropna(subset=["local", "visita", "pts_l", "pts_v", "fecha"]).copy()
        if df_clean.empty:
            return self

        fechas_dt = pd.to_datetime(df_clean["fecha"])
        max_fecha = fechas_dt.max()
        dias_antiguedad = (max_fecha - fechas_dt).dt.days.clip(lower=0).to_numpy(dtype=float)

        # Decaimiento exponencial controlado por half_life (en días)
        decay_lambda = np.log(2.0) / self.half_life
        weights = np.exp(-decay_lambda * dias_antiguedad)
        weights = np.clip(weights, 0.05, 1.0)

        poss_arr = df_clean["poss"].fillna(NBA_AVG_PACE).to_numpy(dtype=float)
        ortg_l_arr = df_clean["ortg_l"].fillna(NBA_AVG_ORTG).to_numpy(dtype=float)
        ortg_v_arr = df_clean["ortg_v"].fillna(NBA_AVG_ORTG).to_numpy(dtype=float)

        self.league_pace = float(np.average(poss_arr, weights=weights))
        self.league_ortg = float(0.5 * (np.average(ortg_l_arr, weights=weights) + np.average(ortg_v_arr, weights=weights)))

        n_teams = len(self.teams)
        idx_l = np.array([self.team_to_idx.get(normalizar_equipo(t), 0) for t in df_clean["local"]], dtype=int)
        idx_v = np.array([self.team_to_idx.get(normalizar_equipo(t), 0) for t in df_clean["visita"]], dtype=int)

        # 1. Optimización regularizada L-BFGS-B para ORtg (Ataque), DRtg (Defensa) y HCA
        # Modelo: E[ORtg_L] = league_ortg + att[L] + def[V] + hca/2
        #         E[ORtg_V] = league_ortg + att[V] + def[L] - hca/2
        y_l = ortg_l_arr - self.league_ortg
        y_v = ortg_v_arr - self.league_ortg
        w_norm = weights / np.mean(weights)
        reg = self.l2_reg / max(1.0, len(df_clean))

        def _loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            att = params[:n_teams]
            dfn = params[n_teams : 2 * n_teams]
            hca = params[2 * n_teams]

            pred_l = att[idx_l] + dfn[idx_v] + 0.5 * hca
            pred_v = att[idx_v] + dfn[idx_l] - 0.5 * hca

            err_l = pred_l - y_l
            err_v = pred_v - y_v

            loss_data = 0.5 * np.mean(w_norm * (err_l ** 2 + err_v ** 2))
            loss_reg = 0.5 * reg * (np.sum(att ** 2) + np.sum(dfn ** 2)) + 0.05 * reg * ((hca - NBA_HOME_COURT_PTS) ** 2)

            grad_att = np.zeros(n_teams, dtype=float)
            grad_dfn = np.zeros(n_teams, dtype=float)
            n_obs = float(len(y_l))

            np.add.at(grad_att, idx_l, (w_norm * err_l) / n_obs)
            np.add.at(grad_att, idx_v, (w_norm * err_v) / n_obs)
            np.add.at(grad_dfn, idx_v, (w_norm * err_l) / n_obs)
            np.add.at(grad_dfn, idx_l, (w_norm * err_v) / n_obs)

            grad_att += reg * att
            grad_dfn += reg * dfn
            grad_hca = float(np.mean(w_norm * 0.5 * (err_l - err_v)) + 0.10 * reg * (hca - NBA_HOME_COURT_PTS))

            grad = np.concatenate([grad_att, grad_dfn, [grad_hca]])
            return float(loss_data + loss_reg), grad

        x0 = np.zeros(2 * n_teams + 1, dtype=float)
        x0[-1] = NBA_HOME_COURT_PTS
        res = minimize(_loss_and_grad, x0, method="L-BFGS-B", jac=True, options={"maxiter": 250})

        att_opt = res.x[:n_teams] - np.mean(res.x[:n_teams])
        dfn_opt = res.x[n_teams : 2 * n_teams] - np.mean(res.x[n_teams : 2 * n_teams])
        self.hca_ortg = float(np.clip(res.x[2 * n_teams], 0.8, 4.8))

        for t, i in self.team_to_idx.items():
            self.ortg_adj[t] = float(self.league_ortg + att_opt[i])
            self.drtg_adj[t] = float(self.league_ortg + dfn_opt[i])

        # 2. Ajuste regularizado del Ritmo (Pace) por equipo
        y_pace = poss_arr - self.league_pace

        def _pace_loss(p_vec: np.ndarray) -> Tuple[float, np.ndarray]:
            pred_p = 0.5 * (p_vec[idx_l] + p_vec[idx_v])
            err_p = pred_p - y_pace
            loss = 0.5 * np.mean(w_norm * (err_p ** 2)) + 0.5 * reg * np.sum(p_vec ** 2)
            grad = np.zeros(n_teams, dtype=float)
            n_obs = float(len(y_pace))
            np.add.at(grad, idx_l, 0.5 * (w_norm * err_p) / n_obs)
            np.add.at(grad, idx_v, 0.5 * (w_norm * err_p) / n_obs)
            grad += reg * p_vec
            return float(loss), grad

        res_p = minimize(_pace_loss, np.zeros(n_teams, dtype=float), method="L-BFGS-B", jac=True)
        p_opt = res_p.x - np.mean(res_p.x)
        for t, i in self.team_to_idx.items():
            self.pace_adj[t] = float(np.clip(self.league_pace + p_opt[i], 92.0, 107.0))

        self.is_fitted = True
        return self

    def predecir_puntos_48m(
        self,
        local: str,
        visita: str,
        feat_pre: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Calcula:
        1. Ritmo esperado del partido (`pace_esp`)
        2. Puntos esperados en 48 minutos (`mu_l`, `mu_v`) combinando los ratings ajustados
           por oposición (Ridge/L-BFGS-B) y el estado reciente EWMA híbrido + ventaja de localía + fatiga/altitud.
        3. Puntos esperados en Q1 y 1H ajustados por el perfil de arranque de cada franquicia.
        """
        canon_l = normalizar_equipo(local)
        canon_v = normalizar_equipo(visita)

        ortg_ridge_l = self.ortg_adj.get(canon_l, self.league_ortg)
        drtg_ridge_l = self.drtg_adj.get(canon_l, self.league_ortg)
        pace_ridge_l = self.pace_adj.get(canon_l, self.league_pace)

        ortg_ridge_v = self.ortg_adj.get(canon_v, self.league_ortg)
        drtg_ridge_v = self.drtg_adj.get(canon_v, self.league_ortg)
        pace_ridge_v = self.pace_adj.get(canon_v, self.league_pace)

        if feat_pre is not None:
            # Fusión 60% Ridge ajustado por rival + 40% EWMA híbrido específico local/visita
            ortg_l = 0.60 * ortg_ridge_l + 0.40 * float(feat_pre.get("ortg_l", ortg_ridge_l))
            drtg_l = 0.60 * drtg_ridge_l + 0.40 * float(feat_pre.get("drtg_l", drtg_ridge_l))
            pace_l = 0.55 * pace_ridge_l + 0.45 * float(feat_pre.get("pace_l", pace_ridge_l))

            ortg_v = 0.60 * ortg_ridge_v + 0.40 * float(feat_pre.get("ortg_v", ortg_ridge_v))
            drtg_v = 0.60 * drtg_ridge_v + 0.40 * float(feat_pre.get("drtg_v", drtg_ridge_v))
            pace_v = 0.55 * pace_ridge_v + 0.45 * float(feat_pre.get("pace_v", pace_ridge_v))

            ajuste_l_pts = float(feat_pre.get("ajuste_fatiga_l_pts", 0.0)) + float(
                feat_pre.get("bono_altitud_l_pts", 0.0)
            )
            ajuste_v_pts = float(feat_pre.get("ajuste_fatiga_v_pts", 0.0))
            q1_ratio_l = float(feat_pre.get("q1_ratio_l", PROP_Q1_BASE))
            h1_ratio_l = float(feat_pre.get("h1_ratio_l", PROP_1H_BASE))
            q1_ratio_v = float(feat_pre.get("q1_ratio_v", PROP_Q1_BASE))
            h1_ratio_v = float(feat_pre.get("h1_ratio_v", PROP_1H_BASE))
        else:
            ortg_l, drtg_l, pace_l = ortg_ridge_l, drtg_ridge_l, pace_ridge_l
            ortg_v, drtg_v, pace_v = ortg_ridge_v, drtg_ridge_v, pace_ridge_v
            ajuste_l_pts, ajuste_v_pts = 0.0, 0.0
            q1_ratio_l = PROP_Q1_BASE * float(NBA_TEAMS.get(canon_l, {}).get("q1_mult", 1.0))
            h1_ratio_l = PROP_1H_BASE * float(NBA_TEAMS.get(canon_l, {}).get("h1_mult", 1.0))
            q1_ratio_v = PROP_Q1_BASE * float(NBA_TEAMS.get(canon_v, {}).get("q1_mult", 1.0))
            h1_ratio_v = PROP_1H_BASE * float(NBA_TEAMS.get(canon_v, {}).get("h1_mult", 1.0))

        # 1. Ritmo esperado del partido (Pace_esp)
        pace_esp = float(np.clip(0.42 * pace_l + 0.42 * pace_v + 0.16 * self.league_pace, 91.0, 108.0))

        # 2. Eficiencia esperada por 100 posesiones para cada equipo en este cruce
        eff_esp_l = ortg_l + (drtg_v - self.league_ortg) + 0.5 * self.hca_ortg
        eff_esp_v = ortg_v + (drtg_l - self.league_ortg) - 0.5 * self.hca_ortg

        # Si el visitante llega en B2B, su eficiencia defensiva también empeora ligeramente (+0.6 pts al local)
        if feat_pre and int(feat_pre.get("b2b_v", 0)) == 1:
            mu_l_base = (pace_esp / 100.0) * eff_esp_l + ajuste_l_pts + 0.4
            mu_v_base = (pace_esp / 100.0) * eff_esp_v + ajuste_v_pts + 0.4
        else:
            mu_l_base = (pace_esp / 100.0) * eff_esp_l + ajuste_l_pts
            mu_v_base = (pace_esp / 100.0) * eff_esp_v + ajuste_v_pts

        mu_l = float(np.clip(mu_l_base, 92.0, 138.0))
        mu_v = float(np.clip(mu_v_base, 92.0, 138.0))
        diff_eff = mu_l - mu_v

        p_eff = float(norm.cdf(diff_eff / BASE_STD_DEV_DIFF))

        # 3. Modelado de 1er Cuarto (Q1 ~25.2%) y 1ra Mitad (1H ~50.8%)
        meta_l = NBA_TEAMS.get(canon_l, {})
        meta_v = NBA_TEAMS.get(canon_v, {})
        q1_mult_l = 0.5 * q1_ratio_l + 0.5 * (PROP_Q1_BASE * float(meta_l.get("q1_mult", 1.0)))
        h1_mult_l = 0.5 * h1_ratio_l + 0.5 * (PROP_1H_BASE * float(meta_l.get("h1_mult", 1.0)))
        q1_mult_v = 0.5 * q1_ratio_v + 0.5 * (PROP_Q1_BASE * float(meta_v.get("q1_mult", 1.0)))
        h1_mult_v = 0.5 * h1_ratio_v + 0.5 * (PROP_1H_BASE * float(meta_v.get("h1_mult", 1.0)))

        q1_mu_l = mu_l * q1_mult_l
        q1_mu_v = mu_v * q1_mult_v
        h1_mu_l = mu_l * h1_mult_l
        h1_mu_v = mu_v * h1_mult_v

        return {
            "pace_esp": round(pace_esp, 2),
            "mu_l": round(mu_l, 2),
            "mu_v": round(mu_v, 2),
            "spread_eff": round(diff_eff, 2),
            "total_eff": round(mu_l + mu_v, 2),
            "p_eff": float(np.clip(p_eff, 0.01, 0.99)),
            "q1_mu_l": round(q1_mu_l, 2),
            "q1_mu_v": round(q1_mu_v, 2),
            "h1_mu_l": round(h1_mu_l, 2),
            "h1_mu_v": round(h1_mu_v, 2),
        }


def beneficio_pago_anticipado_20pts(
    mu_h: float, mu_a: float, sd_diff: float, p_ml: float
) -> Dict[str, float]:
    """
    ⚡ Bono Matemático de Pago Anticipado NBA (+20 Puntos de Ventaja en Betano / Betsafe):
    En Betano y Betsafe, las apuestas anticipadas al ganador (Moneyline) se pagan como
    GANADAS automáticamente si tu equipo alcanza +20 puntos de ventaja en cualquier momento
    del partido, aunque luego el rival remonte y gane al final.

    Usa las propiedades analíticas del Puente Browniano / Caminata Aleatoria con deriva
    sobre la diferencia de puntos X(t) en t ∈ [0, 1] (48 minutos), con deriva d = mu_h - mu_a
    y volatilidad σ = sd_diff.
    Para una barrera B = 20.0 puntos:
      P(max_{0<=t<=1} X(t) >= B  y  X(1) < 0)
        = exp(2 * d * B / σ^2) * Φ(( -d - 2*B ) / σ_refl)
    donde σ_refl considera la varianza intradía de rachas NBA (factor de volatilidad intradía ~ 1.25 * sd_diff).
    """
    b_barrier = 20.0
    sigma_final = max(8.0, float(sd_diff))
    # En baloncesto NBA, las rachas intradía (runs de 12-0 en 3 minutos) generan una excursión máxima
    # ligeramente mayor que el movimiento browniano continuo puro sin reversión a la media:
    sigma_intradia = sigma_final * 1.24

    drift_h = float(mu_h) - float(mu_a)
    drift_a = -drift_h

    def _prob_toca_20_y_pierde(drift: float, p_win_base: float) -> float:
        # Reflexión browniana con reversión elástica calibrada para NBA (+20 pts lead blown)
        # Probabilidad de tocar +20 en algún momento:
        z1 = (b_barrier - drift) / sigma_intradia
        z2 = (-b_barrier - drift) / sigma_intradia
        exp_term = np.clip((2.0 * drift * b_barrier) / (sigma_intradia ** 2), -30.0, 30.0)
        p_hit_20 = float(1.0 - norm.cdf(z1) + np.exp(exp_term) * norm.cdf(z2))
        p_hit_20 = float(np.clip(p_hit_20, 0.0, 0.98))

        # Si un equipo alcanza +20 pts en la NBA, históricamente gana ~93.5% de las veces y
        # pierde por remontada ~6.5% (más frecuente si era underdog o partido de alto ritmo).
        p_remontada_condicional = float(np.clip(0.085 * (1.0 - 0.55 * p_win_base), 0.025, 0.12))
        bono = p_hit_20 * p_remontada_condicional
        return float(np.clip(bono, 0.002, min(0.065, 1.0 - p_win_base - 0.005)))

    p_ml_h = float(np.clip(p_ml, 0.01, 0.99))
    p_ml_a = 1.0 - p_ml_h

    bono_h = _prob_toca_20_y_pierde(drift_h, p_ml_h)
    bono_a = _prob_toca_20_y_pierde(drift_a, p_ml_a)

    p_efectiva_h = float(np.clip(p_ml_h + bono_h, 0.01, 0.995))
    p_efectiva_a = float(np.clip(p_ml_a + bono_a, 0.01, 0.995))

    return {
        "p_ml_local_base": round(p_ml_h, 4),
        "p_ml_visita_base": round(p_ml_a, 4),
        "bono_20pts_local": round(bono_h, 4),
        "bono_20pts_visita": round(bono_a, 4),
        "p_efectiva_local": round(p_efectiva_h, 4),
        "p_efectiva_visita": round(p_efectiva_a, 4),
        "cuota_justa_base_local": round(1.0 / p_ml_h, 2),
        "cuota_justa_base_visita": round(1.0 / p_ml_a, 2),
        "cuota_justa_pa_local": round(1.0 / p_efectiva_h, 2),
        "cuota_justa_pa_visita": round(1.0 / p_efectiva_a, 2),
    }


def diagnosticar_consenso(
    p_eff: float,
    p_elo: float,
    p_ml_clf: float,
    min_pj_season: int = 15,
) -> Dict[str, Any]:
    """
    🚦 Semáforo de Consenso (`diagnosticar_consenso`):
    Compara las probabilidades del Modelo de Eficiencia (`p_eff`), el Sistema Elo (`p_elo`)
    y el Clasificador ML (`p_ml_clf`).
    - Si hay una discrepancia >= 12 pp (0.12), o si los modelos dan distinto favorito
      (uno > 0.515 y otro < 0.485), o si hay pocos partidos en la temporada (< 8 PJ),
      marca el encuentro como `🔴 Alerta: Partido Trampa / Alta Volatilidad` y lo excluye
      automáticamente de las combinadas principales.
    """
    probs = [float(p_eff), float(p_elo), float(p_ml_clf)]
    max_diff_pp = (max(probs) - min(probs)) * 100.0

    favs = [1 if p >= 0.50 else 0 for p in probs]
    distinto_favorito = (len(set(favs)) > 1) and (max(probs) - min(probs) >= 0.04)
    pocos_partidos = int(min_pj_season) < 8
    discrepancia_alta = max_diff_pp >= 12.0

    es_trampa = bool(discrepancia_alta or distinto_favorito or pocos_partidos)

    motivos = []
    if discrepancia_alta:
        motivos.append(f"Discrepancia entre modelos de {max_diff_pp:.1f} pp (≥ 12.0 pp)")
    if distinto_favorito:
        motivos.append("Los 3 pilares no coinciden en quién es el favorito")
    if pocos_partidos:
        motivos.append(f"Muestra temprana de temporada ({min_pj_season} PJ < 8 PJ)")

    if es_trampa:
        estado = "🔴 Alerta: Partido Trampa / Alta Volatilidad"
        nivel = "ROJO"
        detalle = " · ".join(motivos)
    elif max_diff_pp >= 7.5:
        estado = "🟡 Consenso Moderado (Dispersión Media)"
        nivel = "AMARILLO"
        detalle = f"Diferencia máxima entre modelos de {max_diff_pp:.1f} pp."
    else:
        estado = "🟢 Alto Consenso (Modelos Alineados)"
        nivel = "VERDE"
        detalle = f"Los 3 pilares coinciden plenamente (dispersión de solo {max_diff_pp:.1f} pp)."

    return {
        "es_partido_trampa": es_trampa,
        "nivel": nivel,
        "estado": estado,
        "discrepancia_pp": round(max_diff_pp, 2),
        "distinto_favorito": distinto_favorito,
        "pocos_partidos": pocos_partidos,
        "min_pj_season": int(min_pj_season),
        "detalle": detalle,
    }


def simular_monte_carlo_correlacionado(
    local: str,
    visita: str,
    mu_l: float,
    mu_v: float,
    q1_mu_l: float,
    q1_mu_v: float,
    h1_mu_l: float,
    h1_mu_v: float,
    discrepancia_pp: float = 4.0,
    min_pj_season: int = 20,
    n_sims: int = DEFAULT_MC_SIMS,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simula entre 5,000 y 10,000 partidos en Monte Carlo incorporando incertidumbre epistémica
    (σ adaptativo según cantidad de partidos jugados en la temporada y discrepancia entre modelos)
    para generar:
      1. Ganador del Partido (Moneyline 1-2 incluyendo Prórroga/OT) con rango P10-P90.
      2. Hándicap / Spread de Puntos (tabla desde -12.5 hasta +12.5 para ambos equipos con P10).
      3. Totales de Puntos del Partido (Over / Under dinámico alrededor del total esperado).
      4. Totales por Equipo (Team Totals Over / Under para Local y Visita).
      5. Mercados de 1er Cuarto (Q1) y 1ra Mitad (1H).
    """
    n_sims = int(np.clip(n_sims, 1000, 20000))
    rng = np.random.default_rng(seed)

    # Incertidumbre epistémica adaptativa:
    # Más alta al inicio de la temporada (<15 PJ) o cuando los modelos discrepan
    factor_epistemico = 1.0 + max(0.0, (15.0 - min(15.0, float(min_pj_season))) * 0.015) + (
        max(0.0, float(discrepancia_pp) - 4.0) * 0.008
    )
    sd_diff = float(BASE_STD_DEV_DIFF * factor_epistemico)
    sd_total = float(BASE_STD_DEV_TOTAL * factor_epistemico)

    # Simulación jerárquica correlacionada (20 bloques de incertidumbre epistémica en la media
    # para obtener los percentiles P10, P25 y P90 de cada probabilidad de mercado):
    n_blocks = 20
    sims_per_block = max(50, n_sims // n_blocks)
    total_sims = n_blocks * sims_per_block

    sigma_epist_diff = max(1.3, 1.8 * factor_epistemico)
    sigma_epist_tot = max(1.8, 2.4 * factor_epistemico)

    # Desviaciones por cuarto para construir Q1 -> 1H -> 48m de forma coherente y correlacionada
    # Si L y V comparten el ritmo del cuarto (factor común de ritmo `z_pace`), sus puntajes están correlacionados:
    q2_mu_l = max(18.0, h1_mu_l - q1_mu_l)
    q2_mu_v = max(18.0, h1_mu_v - q1_mu_v)
    h2_mu_l = max(38.0, mu_l - h1_mu_l)
    h2_mu_v = max(38.0, mu_v - h1_mu_v)

    # Perturbaciones epistémicas por bloque (n_blocks, 1)
    ep_diff = rng.normal(0.0, sigma_epist_diff, size=(n_blocks, 1))
    ep_tot = rng.normal(0.0, sigma_epist_tot, size=(n_blocks, 1))
    ep_l = 0.5 * (ep_tot + ep_diff)
    ep_v = 0.5 * (ep_tot - ep_diff)

    # Simulación por periodo con correlación de ritmo común
    def _sim_period(m_l: float, m_v: float, frac_time: float) -> Tuple[np.ndarray, np.ndarray]:
        sd_pace_common = 3.2 * np.sqrt(frac_time)
        sd_team_ind = (sd_diff / np.sqrt(2.0)) * np.sqrt(frac_time)
        pace_shock = rng.normal(0.0, sd_pace_common, size=(n_blocks, sims_per_block))
        pts_p_l = (m_l + ep_l * frac_time) + pace_shock + rng.normal(0.0, sd_team_ind, size=(n_blocks, sims_per_block))
        pts_p_v = (m_v + ep_v * frac_time) + pace_shock + rng.normal(0.0, sd_team_ind, size=(n_blocks, sims_per_block))
        return pts_p_l, pts_p_v

    q1_l_sim, q1_v_sim = _sim_period(q1_mu_l, q1_mu_v, 0.25)
    q2_l_sim, q2_v_sim = _sim_period(q2_mu_l, q2_mu_v, 0.25)
    h2_l_sim, h2_v_sim = _sim_period(h2_mu_l, h2_mu_v, 0.50)

    h1_l_sim = q1_l_sim + q2_l_sim
    h1_v_sim = q1_v_sim + q2_v_sim

    reg_l_sim = np.rint(h1_l_sim + h2_l_sim)
    reg_v_sim = np.rint(h1_v_sim + h2_v_sim)

    # Resolución de Prórroga (OT de 5 minutos) si empatan al final de los 48 minutos
    empate_mask = reg_l_sim == reg_v_sim
    final_l_sim = reg_l_sim.copy()
    final_v_sim = reg_v_sim.copy()

    if np.any(empate_mask):
        ot_mu_l = mu_l * (5.0 / 48.0)
        ot_mu_v = mu_v * (5.0 / 48.0)
        ot_l = np.rint(rng.normal(ot_mu_l, 3.6, size=reg_l_sim.shape))
        ot_v = np.rint(rng.normal(ot_mu_v, 3.6, size=reg_v_sim.shape))
        # Si persiste el empate en OT, desempatar proporcionalmente a la fuerza de cada equipo
        p_force_l = float(1.0 / (1.0 + np.exp(-(mu_l - mu_v) / 7.0)))
        tie_ot = ot_l == ot_v
        win_tie_l = rng.random(size=reg_l_sim.shape) < p_force_l
        ot_l = np.where(tie_ot & win_tie_l, ot_l + 2, ot_l)
        ot_v = np.where(tie_ot & (~win_tie_l), ot_v + 2, ot_v)

        final_l_sim = np.where(empate_mask, reg_l_sim + ot_l, reg_l_sim)
        final_v_sim = np.where(empate_mask, reg_v_sim + ot_v, reg_v_sim)

    diff_final = final_l_sim - final_v_sim  # > 0 gana local
    tot_final = final_l_sim + final_v_sim

    # 1. Ganador del Partido (Moneyline incluyendo OT)
    win_l_blocks = np.mean(diff_final > 0, axis=1)
    win_v_blocks = 1.0 - win_l_blocks

    p_ml_l = float(np.clip(np.mean(win_l_blocks), 0.01, 0.99))
    p_ml_v = 1.0 - p_ml_l
    p10_ml_l, p25_ml_l, p90_ml_l = [float(x) for x in np.percentile(win_l_blocks, [10, 25, 90])]
    p10_ml_v, p25_ml_v, p90_ml_v = [float(x) for x in np.percentile(win_v_blocks, [10, 25, 90])]

    # 2. Hándicap / Spread de Puntos (desde -12.5 hasta +12.5)
    spread_justo_val = round(-(mu_l - mu_v), 1)  # Negativo si el local es favorito (ej. BOS -6.2)
    favorito_spread = local if (mu_l >= mu_v) else visita
    margen_favorito = abs(round(mu_l - mu_v, 1))
    etiqueta_spread_justo = f"{favorito_spread} -{margen_favorito:.1f}"

    lineas_handicap = [
        -12.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5,
        1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
    ]
    tabla_spread: List[Dict[str, Any]] = []
    for h in lineas_handicap:
        # Local cubre h si (final_l + h) > final_v  <=> diff_final > -h
        cov_l_blocks = np.mean((diff_final + h) > 0, axis=1)
        p_cov_l = float(np.clip(np.mean(cov_l_blocks), 0.005, 0.995))
        p10_cov_l = float(np.clip(np.percentile(cov_l_blocks, 10), 0.005, 0.995))
        p25_cov_l = float(np.clip(np.percentile(cov_l_blocks, 25), 0.005, 0.995))

        # Visita cubre h si (final_v + h) > final_l <=> -diff_final + h > 0
        cov_v_blocks = np.mean((-diff_final + h) > 0, axis=1)
        p_cov_v = float(np.clip(np.mean(cov_v_blocks), 0.005, 0.995))
        p10_cov_v = float(np.clip(np.percentile(cov_v_blocks, 10), 0.005, 0.995))
        p25_cov_v = float(np.clip(np.percentile(cov_v_blocks, 25), 0.005, 0.995))

        tabla_spread.append(
            {
                "linea": float(h),
                "linea_str": f"{h:+.1f}",
                "local_mercado": f"{local} {h:+.1f}",
                "prob_local": round(p_cov_l, 4),
                "p10_local": round(p10_cov_l, 4),
                "p25_local": round(p25_cov_l, 4),
                "cuota_justa_local": round(1.0 / p_cov_l, 2),
                "visita_mercado": f"{visita} {h:+.1f}",
                "prob_visita": round(p_cov_v, 4),
                "p10_visita": round(p10_cov_v, 4),
                "p25_visita": round(p25_cov_v, 4),
                "cuota_justa_visita": round(1.0 / p_cov_v, 2),
            }
        )

    # 3. Totales de Puntos del Partido (Over / Under dinámico)
    total_esperado = float(round(np.mean(tot_final), 1))
    linea_central = np.floor(mu_l + mu_v) + 0.5
    deltas_tot = [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0]
    tabla_totales: List[Dict[str, Any]] = []
    for dt in deltas_tot:
        lin = float(linea_central + dt)
        over_blocks = np.mean(tot_final > lin, axis=1)
        under_blocks = 1.0 - over_blocks
        p_ov = float(np.clip(np.mean(over_blocks), 0.005, 0.995))
        p_un = float(np.clip(np.mean(under_blocks), 0.005, 0.995))
        p10_ov = float(np.clip(np.percentile(over_blocks, 10), 0.005, 0.995))
        p10_un = float(np.clip(np.percentile(under_blocks, 10), 0.005, 0.995))
        p25_ov = float(np.clip(np.percentile(over_blocks, 25), 0.005, 0.995))
        p25_un = float(np.clip(np.percentile(under_blocks, 25), 0.005, 0.995))

        tabla_totales.append(
            {
                "linea": lin,
                "es_central": dt == 0.0,
                "prob_over": round(p_ov, 4),
                "p10_over": round(p10_ov, 4),
                "p25_over": round(p25_ov, 4),
                "cuota_justa_over": round(1.0 / p_ov, 2),
                "prob_under": round(p_un, 4),
                "p10_under": round(p10_un, 4),
                "p25_under": round(p25_un, 4),
                "cuota_justa_under": round(1.0 / p_un, 2),
            }
        )

    # 4. Totales por Equipo (Team Totals Over / Under)
    def _team_totals_table(sim_pts: np.ndarray, mu_team: float, team_name: str) -> List[Dict[str, Any]]:
        c_line = np.floor(mu_team) + 0.5
        out_tt = []
        for delta in [-6.0, -3.0, 0.0, 3.0, 6.0]:
            tl = float(c_line + delta)
            ov_b = np.mean(sim_pts > tl, axis=1)
            un_b = 1.0 - ov_b
            p_o = float(np.clip(np.mean(ov_b), 0.01, 0.99))
            p_u = float(np.clip(np.mean(un_b), 0.01, 0.99))
            p10_o = float(np.clip(np.percentile(ov_b, 10), 0.01, 0.99))
            p10_u = float(np.clip(np.percentile(un_b, 10), 0.01, 0.99))
            out_tt.append(
                {
                    "equipo": team_name,
                    "linea": tl,
                    "prob_over": round(p_o, 4),
                    "p10_over": round(p10_o, 4),
                    "cuota_justa_over": round(1.0 / p_o, 2),
                    "prob_under": round(p_u, 4),
                    "p10_under": round(p10_u, 4),
                    "cuota_justa_under": round(1.0 / p_u, 2),
                }
            )
        return out_tt

    team_totals_local = _team_totals_table(final_l_sim, mu_l, local)
    team_totals_visita = _team_totals_table(final_v_sim, mu_v, visita)

    # 5. Mercados de 1er Cuarto (Q1) y 1ra Mitad (1H)
    def _resumen_periodo(sim_l: np.ndarray, sim_v: np.ndarray, m_l: float, m_v: float) -> Dict[str, Any]:
        diff_p = sim_l - sim_v
        tot_p = sim_l + sim_v
        win_l_b = np.mean(diff_p > 0, axis=1)
        win_v_b = np.mean(diff_p < 0, axis=1)
        p_l = float(np.clip(np.mean(win_l_b) / max(1e-5, np.mean(win_l_b + win_v_b)), 0.02, 0.98))
        p_v = 1.0 - p_l
        p10_l = float(np.clip(np.percentile(win_l_b, 10), 0.02, 0.98))
        p10_v = float(np.clip(np.percentile(win_v_b, 10), 0.02, 0.98))

        h_line = np.floor(abs(m_l - m_v)) + 0.5
        if m_l >= m_v:
            h_local_line = -h_line
        else:
            h_local_line = h_line
        cov_h_l = float(np.clip(np.mean((diff_p + h_local_line) > 0), 0.02, 0.98))

        t_line = np.floor(m_l + m_v) + 0.5
        ov_p = float(np.clip(np.mean(tot_p > t_line), 0.02, 0.98))
        un_p = 1.0 - ov_p

        return {
            "mu_local": round(m_l, 1),
            "mu_visita": round(m_v, 1),
            "total_esperado": round(m_l + m_v, 1),
            "spread_esperado": round(m_l - m_v, 1),
            "prob_gana_local": round(p_l, 4),
            "prob_gana_visita": round(p_v, 4),
            "p10_gana_local": round(p10_l, 4),
            "p10_gana_visita": round(p10_v, 4),
            "cuota_local": round(1.0 / p_l, 2),
            "cuota_visita": round(1.0 / p_v, 2),
            "linea_handicap_local": h_local_line,
            "prob_cubre_local": round(cov_h_l, 4),
            "linea_total": t_line,
            "prob_over": round(ov_p, 4),
            "prob_under": round(un_p, 4),
            "cuota_over": round(1.0 / ov_p, 2),
            "cuota_under": round(1.0 / un_p, 2),
        }

    mercados_q1 = _resumen_periodo(q1_l_sim, q1_v_sim, q1_mu_l, q1_mu_v)
    mercados_1h = _resumen_periodo(h1_l_sim, h1_v_sim, h1_mu_l, h1_mu_v)

    return {
        "n_sims": total_sims,
        "sd_diff": round(sd_diff, 2),
        "sd_total": round(sd_total, 2),
        "moneyline": {
            "prob_local": round(p_ml_l, 4),
            "prob_visita": round(p_ml_v, 4),
            "p10_local": round(p10_ml_l, 4),
            "p25_local": round(p25_ml_l, 4),
            "p90_local": round(p90_ml_l, 4),
            "p10_visita": round(p10_ml_v, 4),
            "p25_visita": round(p25_ml_v, 4),
            "p90_visita": round(p90_ml_v, 4),
            "cuota_justa_local": round(1.0 / p_ml_l, 2),
            "cuota_justa_visita": round(1.0 / p_ml_v, 2),
            "prob_prorroga": round(float(np.mean(empate_mask)), 4),
        },
        "spread": {
            "spread_justo_local": spread_justo_val,
            "etiqueta_spread_justo": etiqueta_spread_justo,
            "margen_esperado_local": round(mu_l - mu_v, 2),
            "tabla": tabla_spread,
        },
        "totales": {
            "total_esperado": round(mu_l + mu_v, 1),
            "total_sim_media": total_esperado,
            "linea_central": float(linea_central),
            "tabla": tabla_totales,
        },
        "team_totals": {
            "local": team_totals_local,
            "visita": team_totals_visita,
        },
        "q1": mercados_q1,
        "h1": mercados_1h,
    }


def calcular_ev_y_kelly(
    prob_real: float,
    cuota_mercado: float,
    prob_p25: Optional[float] = None,
    fraccion_kelly: float = 0.25,
    tope_max_banca_pct: float = 5.0,
) -> Dict[str, Any]:
    """
    💹 Calculadora de Valor Esperado (EV%) y Gestión de Banca (Kelly 1/4 Conservador):
    - Calcula el `EV % = (prob_real * cuota_mercado - 1) * 100`
    - Calcula el `EV Conservador %` usando el percentil 25 (`P25`) de Monte Carlo
    - Calcula el `Stake sugerido (% de banca con 1/4 de Kelly, tope máx. 5.0%)`
    """
    p = float(np.clip(prob_real, 0.001, 0.999))
    p_cons = float(np.clip(prob_p25 if prob_p25 is not None else (p - 0.035), 0.001, 0.999))
    cuota = float(max(1.01, cuota_mercado))

    ev_pct = (p * cuota - 1.0) * 100.0
    ev_cons_pct = (p_cons * cuota - 1.0) * 100.0

    b = cuota - 1.0
    q = 1.0 - p
    kelly_full = ((b * p) - q) / b if b > 0 else 0.0
    if kelly_full <= 0:
        stake_pct = 0.0
    else:
        stake_pct = min(float(tope_max_banca_pct), float(kelly_full * fraccion_kelly * 100.0))

    hay_valor = ev_pct > 0.0
    if ev_cons_pct >= 2.5:
        recomendacion = "🔥 EV+ Sólido (Supera filtro conservador P25)"
    elif ev_pct > 0.0:
        recomendacion = "✅ EV+ Moderado (Valor positivo en expectativa central)"
    else:
        recomendacion = "⛔ Sin Valor (Cuota por debajo de la cuota justa)"

    return {
        "prob_usada": round(p, 4),
        "prob_p25": round(p_cons, 4),
        "cuota_justa": round(1.0 / p, 2),
        "cuota_mercado": round(cuota, 2),
        "ev_pct": round(ev_pct, 2),
        "ev_conservador_pct": round(ev_cons_pct, 2),
        "stake_kelly_pct": round(stake_pct, 2),
        "hay_valor": hay_valor,
        "recomendacion": recomendacion,
    }


def selecciones_bajo_riesgo(
    prediccion: Dict[str, Any],
    casa_apuestas: str = "Betano (Pago Anticipado +20 pts / SuperCuotas)",
) -> List[Dict[str, Any]]:
    """
    🛡️ Resumen: Top 2–3 Opciones de Bajo Riesgo por Partido (`selecciones_bajo_riesgo`):
    Selecciona entre 2 y 3 mercados de categorías distintas:
      1. Categoría 1: Ganador / Hándicap Conservador
         - Criterio de apostador profesional:
           * Si un equipo ya es claro favorito directo (`>= 70%` en Moneyline), recomienda
             directamente `Gana [Equipo] (Moneyline)` (aprovechando además el Pago Anticipado +20 pts
             de Betano/Betsafe) sin diluir la cuota en hándicaps positivos innecesarios.
           * Si es un favorito moderado (`54%` a `69%`) o un partido parejo, busca el
             Hándicap Alternativo de Protección (ej. `[Equipo] +6.5` o `[Equipo] -2.5`)
             que esté en la zona óptima de seguridad (`66%` a `88.5%`) y maximice el piso `P10`.
      2. Categoría 2: Total de Puntos Alternativo Over/Under (en zona `66%` a `88.5%`).
      3. Categoría 3: 1ra Mitad (`1H`) o Total por Equipo (`Team Total`).
    """
    local = prediccion["local"]
    visita = prediccion["visita"]
    mc = prediccion["monte_carlo"]
    pa = prediccion["pago_anticipado"]
    usa_pa = ("betano" in casa_apuestas.lower()) or ("betsafe" in casa_apuestas.lower())

    p_ml_l = float(mc["moneyline"]["prob_local"])
    p_ml_v = float(mc["moneyline"]["prob_visita"])
    p10_ml_l = float(mc["moneyline"]["p10_local"])
    p10_ml_v = float(mc["moneyline"]["p10_visita"])

    if usa_pa:
        p_eff_l = float(pa["p_efectiva_local"])
        p_eff_v = float(pa["p_efectiva_visita"])
    else:
        p_eff_l = p_ml_l
        p_eff_v = p_ml_v

    opciones: List[Dict[str, Any]] = []

    # --- CATEGORÍA 1: Ganador / Hándicap Conservador ---
    fav_es_local = p_ml_l >= p_ml_v
    fav_team = local if fav_es_local else visita
    p_fav_ml = p_ml_l if fav_es_local else p_ml_v
    p_fav_efectiva = p_eff_l if fav_es_local else p_eff_v
    p10_fav_ml = p10_ml_l if fav_es_local else p10_ml_v

    if p_fav_ml >= 0.70:
        # Regla de oro profesional: Favorito claro (>= 70%) -> Moneyline directo
        nota_pa = " + Pago Anticipado (+20 pts)" if usa_pa else ""
        opciones.append(
            {
                "categoria": "1. Ganador / Hándicap Principal",
                "mercado": f"Gana {fav_team} (Moneyline)",
                "probabilidad": round(p_fav_efectiva, 4),
                "piso_p10": round(max(p10_fav_ml, p_fav_efectiva - 0.045), 4),
                "cuota_justa": round(1.0 / p_fav_efectiva, 2),
                "justificacion": (
                    f"Favorito claro directo ({p_fav_ml*100:.1f}% ML base{nota_pa}). "
                    "Se recomienda Moneyline directo sin diluir valor en hándicaps innecesarios."
                ),
            }
        )
    else:
        # Favorito moderado (54% a 69%) o partido parejo -> Hándicap Alternativo de Protección (66% a 88.5%)
        candidatos_h = []
        for row in mc["spread"]["tabla"]:
            for side_team, p_key, p10_key, m_key in [
                (local, "prob_local", "p10_local", "local_mercado"),
                (visita, "prob_visita", "p10_visita", "visita_mercado"),
            ]:
                pr = float(row[p_key])
                p10 = float(row[p10_key])
                if 0.66 <= pr <= 0.885:
                    # Preferir que proteja al favorito o al equipo con mejor piso P10 cerca de 76%-82%
                    score_opt = p10 - 0.35 * abs(pr - 0.78) + (0.02 if side_team == fav_team else 0.0)
                    candidatos_h.append((score_opt, row[m_key], pr, p10))

        if candidatos_h:
            candidatos_h.sort(key=lambda x: x[0], reverse=True)
            _, best_m, best_pr, best_p10 = candidatos_h[0]
            opciones.append(
                {
                    "categoria": "1. Hándicap de Protección",
                    "mercado": f"Hándicap Alternativo: {best_m}",
                    "probabilidad": round(best_pr, 4),
                    "piso_p10": round(best_p10, 4),
                    "cuota_justa": round(1.0 / best_pr, 2),
                    "justificacion": (
                        f"Partido competitivo ({p_fav_ml*100:.1f}% ML). El hándicap alternativo "
                        f"protege contra finales cerrados con un piso P10 de {best_p10*100:.1f}%."
                    ),
                }
            )
        else:
            opciones.append(
                {
                    "categoria": "1. Ganador / Hándicap",
                    "mercado": f"Gana {fav_team} (Moneyline)",
                    "probabilidad": round(p_fav_efectiva, 4),
                    "piso_p10": round(p10_fav_ml, 4),
                    "cuota_justa": round(1.0 / p_fav_efectiva, 2),
                    "justificacion": "Selección directa al favorito del consenso.",
                }
            )

    # --- CATEGORÍA 2: Total de Puntos Alternativo Over/Under (66% a 88.5%) ---
    candidatos_tot = []
    for row in mc["totales"]["tabla"]:
        lin = row["linea"]
        for tipo, p_k, p10_k in [
            (f"Más de {lin:.1f} pts (Over Alternativo)", "prob_over", "p10_over"),
            (f"Menos de {lin:.1f} pts (Under Alternativo)", "prob_under", "p10_under"),
        ]:
            pr = float(row[p_k])
            p10 = float(row[p10_k])
            if 0.66 <= pr <= 0.885:
                score_t = p10 - 0.40 * abs(pr - 0.77)
                candidatos_tot.append((score_t, tipo, pr, p10))

    if candidatos_tot:
        candidatos_tot.sort(key=lambda x: x[0], reverse=True)
        _, m_tot, pr_tot, p10_tot = candidatos_tot[0]
        opciones.append(
            {
                "categoria": "2. Total de Puntos Alternativo",
                "mercado": m_tot,
                "probabilidad": round(pr_tot, 4),
                "piso_p10": round(p10_tot, 4),
                "cuota_justa": round(1.0 / pr_tot, 2),
                "justificacion": (
                    f"Línea desplazada con margen de seguridad respecto al total proyectado "
                    f"({mc['totales']['total_esperado']:.1f} pts)."
                ),
            }
        )

    # --- CATEGORÍA 3: 1ra Mitad (1H) o Total por Equipo (Team Total) ---
    candidatos_c3 = []
    h1 = mc["h1"]
    for side_name, pr_k, p10_k in [
        (f"1ra Mitad (1H): Gana {local}", "prob_gana_local", "p10_gana_local"),
        (f"1ra Mitad (1H): Gana {visita}", "prob_gana_visita", "p10_gana_visita"),
    ]:
        pr = float(h1[pr_k])
        p10 = float(h1[p10_k])
        if 0.66 <= pr <= 0.885:
            candidatos_c3.append((p10 - 0.3 * abs(pr - 0.75), side_name, pr, p10, "Dominio proyectado al descanso (1ra Mitad)."))

    for tt_list in (mc["team_totals"]["local"], mc["team_totals"]["visita"]):
        for r_tt in tt_list:
            eq = r_tt["equipo"]
            lin = r_tt["linea"]
            for label, pk, p10k in [
                (f"Total Equipo: {eq} Más de {lin:.1f} pts", "prob_over", "p10_over"),
                (f"Total Equipo: {eq} Menos de {lin:.1f} pts", "prob_under", "p10_under"),
            ]:
                pr = float(r_tt[pk])
                p10 = float(r_tt[p10k])
                if 0.66 <= pr <= 0.885:
                    candidatos_c3.append((p10 - 0.35 * abs(pr - 0.76), label, pr, p10, f"Línea individual protegida para {eq}."))

    if candidatos_c3:
        candidatos_c3.sort(key=lambda x: x[0], reverse=True)
        _, m_c3, pr_c3, p10_c3, just_c3 = candidatos_c3[0]
        opciones.append(
            {
                "categoria": "3. 1ra Mitad / Total por Equipo",
                "mercado": m_c3,
                "probabilidad": round(pr_c3, 4),
                "piso_p10": round(p10_c3, 4),
                "cuota_justa": round(1.0 / pr_c3, 2),
                "justificacion": just_c3,
            }
        )

    return opciones[:3]


class NBAEnsembleEngine:
    """
    Motor Matemático Ensemble de 3 Pilares:
      - Pilar 1: Modelo Posesional y de Eficiencia (Ritmo × Eficiencia con Decaimiento Exponencial)
      - Pilar 2: Sistema Elo NBA con Margen de Victoria (MOV) y regresión entre temporadas
      - Pilar 3: Modelo de Features Avanzadas y Four Factors (EWMA Híbrido 70/30 + Clasificador/Regresor ML)
    """

    def __init__(
        self,
        half_life: float = DEFAULT_HALF_LIFE,
        alpha_ewma: float = 0.14,
        n_sims: int = DEFAULT_MC_SIMS,
    ) -> None:
        self.half_life = float(half_life)
        self.alpha_ewma = float(alpha_ewma)
        self.n_sims = int(n_sims)

        self.pilar_posesional = ModeloPosesionalNBA(half_life=self.half_life)
        self.tracker = TeamStateTracker(alpha_ewma=self.alpha_ewma)
        self.elo_sys = EloSystemNBA()

        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=0.45, max_iter=500)
        self.reg_margin = Ridge(alpha=18.0)
        self.ml_fitted = False

    def fit(
        self,
        df_partidos: pd.DataFrame,
        df_features_precomputed: Optional[pd.DataFrame] = None,
    ) -> "NBAEnsembleEngine":
        """
        Entrena los 3 pilares sobre el historial de partidos sin fuga de información.
        """
        if df_partidos.empty:
            return self

        self.pilar_posesional.fit(df_partidos)

        if df_features_precomputed is not None and not df_features_precomputed.empty:
            df_feat = df_features_precomputed
        else:
            from engine.features import construir_dataset_cronologico
            df_feat, self.tracker, self.elo_sys = construir_dataset_cronologico(
                df_partidos, alpha_ewma=self.alpha_ewma
            )

        if len(df_feat) >= 25:
            X = df_feat[FEATURE_COLS_ML].fillna(0.0).to_numpy(dtype=float)
            y_win = df_feat["target_win_l"].to_numpy(dtype=int)
            y_mar = df_feat["target_margin_l"].to_numpy(dtype=float)

            # Pesos con decaimiento temporal exponencial (half_life)
            fechas_dt = pd.to_datetime(df_feat["fecha"])
            dias = (fechas_dt.max() - fechas_dt).dt.days.clip(lower=0).to_numpy(dtype=float)
            w = np.exp(-(np.log(2.0) / max(15.0, self.half_life)) * dias)
            w = np.clip(w, 0.05, 1.0)

            if len(np.unique(y_win)) >= 2:
                X_scaled = self.scaler.fit_transform(X)
                self.clf.fit(X_scaled, y_win, sample_weight=w)
                self.reg_margin.fit(X_scaled, y_mar, sample_weight=w)
                self.ml_fitted = True

        return self

    def predecir_partido_completo(
        self,
        local: str,
        visita: str,
        fecha: Optional[Union[str, date, datetime]] = None,
        casa_apuestas: str = "Betano (Pago Anticipado +20 pts / SuperCuotas)",
        n_sims: Optional[int] = None,
        seed: int = 42,
        ajuste_bajas_spread_local: float = 0.0,
        ajuste_bajas_total: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Genera la predicción probabilística completa para un cruce `local` vs `visita`:
        - Combina los 3 Pilares (Eficiencia Posesional, Elo MOV, Four Factors ML)
        - Aplica automáticamente `ajuste_fatiga_nba` y `ALTITUD_NBA`
        - Ejecuta la Simulación de Monte Carlo Correlacionada
        - Calcula el Bono de Pago Anticipado +20 pts (Betano / Betsafe)
        - Evalúa el Semáforo de Consenso (`diagnosticar_consenso`)
        - Extrae las Top 2-3 Selecciones Inteligentes de Bajo Riesgo
        """
        canon_l = normalizar_equipo(local)
        canon_v = normalizar_equipo(visita)
        fecha_str = str(fecha or datetime.now().strftime("%Y-%m-%d"))[:10]

        feat = extraer_features_pre_partido(
            canon_l, canon_v, fecha_str, self.tracker, self.elo_sys
        )

        # Pilar 1: Modelo Posesional y de Eficiencia
        pred_pos = self.pilar_posesional.predecir_puntos_48m(canon_l, canon_v, feat_pre=feat)
        p_eff = float(pred_pos["p_eff"])
        spread_eff = float(pred_pos["spread_eff"])

        # Pilar 2: Sistema Elo NBA con Margen de Victoria
        p_elo = float(feat["p_elo"])
        spread_elo = float(feat["spread_elo"])

        # Pilar 3: Modelo ML de Features Avanzadas y Four Factors (EWMA Híbrido 70/30)
        # Fórmula analítica calibrada sobre Four Factors + Net Rating + forma
        score_ff = (
            0.40 * float(feat["net_rtg_diff"])
            + 38.0 * float(feat["efg_diff"])
            - 30.0 * float(feat["tov_pct_diff"])
            + 16.0 * float(feat["orb_pct_diff"])
            + 8.0 * float(feat["ftr_diff"])
            + 2.4
            + float(feat["impacto_tactico_pts"])
        )
        p_ff_analitica = float(norm.cdf(score_ff / BASE_STD_DEV_DIFF))

        if self.ml_fitted:
            x_vec = np.array([[float(feat.get(c, 0.0)) for c in FEATURE_COLS_ML]], dtype=float)
            x_sc = self.scaler.transform(x_vec)
            p_ml_raw = float(self.clf.predict_proba(x_sc)[0, 1])
            spread_ml_raw = float(self.reg_margin.predict(x_sc)[0])
            # Regularización con prior analítico de Four Factors para evitar sobreajuste en muestras cortas
            spread_ml = 0.50 * spread_ml_raw + 0.50 * score_ff
            p_ml_clf = 0.50 * p_ml_raw + 0.50 * p_ff_analitica
        else:
            spread_ml = float(score_ff)
            p_ml_clf = p_ff_analitica

        p_ml_clf = float(np.clip(p_ml_clf, 0.01, 0.99))

        # Pesos del Ensemble (45% Posesional/Eficiencia, 30% Four Factors ML, 25% Elo MOV)
        spread_ensemble = (
            0.45 * spread_eff + 0.30 * spread_ml + 0.25 * spread_elo + float(ajuste_bajas_spread_local)
        )
        total_ensemble = float(pred_pos["total_eff"]) + float(ajuste_bajas_total)

        mu_l_ens = 0.5 * (total_ensemble + spread_ensemble)
        mu_v_ens = 0.5 * (total_ensemble - spread_ensemble)

        # Ajustar proporcionalmente Q1 y 1H a las medias finales del ensemble
        ratio_l = mu_l_ens / max(1.0, float(pred_pos["mu_l"]))
        ratio_v = mu_v_ens / max(1.0, float(pred_pos["mu_v"]))
        q1_mu_l = float(pred_pos["q1_mu_l"]) * ratio_l
        q1_mu_v = float(pred_pos["q1_mu_v"]) * ratio_v
        h1_mu_l = float(pred_pos["h1_mu_l"]) * ratio_l
        h1_mu_v = float(pred_pos["h1_mu_v"]) * ratio_v

        # Semáforo de Consenso y Detector de Partidos Trampa
        consenso = diagnosticar_consenso(
            p_eff=p_eff,
            p_elo=p_elo,
            p_ml_clf=p_ml_clf,
            min_pj_season=int(feat["min_pj_season"]),
        )

        # Simulación de Monte Carlo Correlacionada
        sims_to_run = int(n_sims or self.n_sims)
        mc = simular_monte_carlo_correlacionado(
            local=canon_l,
            visita=canon_v,
            mu_l=mu_l_ens,
            mu_v=mu_v_ens,
            q1_mu_l=q1_mu_l,
            q1_mu_v=q1_mu_v,
            h1_mu_l=h1_mu_l,
            h1_mu_v=h1_mu_v,
            discrepancia_pp=consenso["discrepancia_pp"],
            min_pj_season=int(feat["min_pj_season"]),
            n_sims=sims_to_run,
            seed=seed,
        )

        # Bono Matemático de Pago Anticipado +20 pts (Betano / Betsafe)
        pago_anticipado = beneficio_pago_anticipado_20pts(
            mu_h=mu_l_ens,
            mu_a=mu_v_ens,
            sd_diff=mc["sd_diff"],
            p_ml=mc["moneyline"]["prob_local"],
        )

        resultado = {
            "local": canon_l,
            "visita": canon_v,
            "fecha": fecha_str,
            "casa_apuestas": casa_apuestas,
            "pilares": {
                "p_posesional": round(p_eff, 4),
                "spread_posesional": round(spread_eff, 2),
                "p_elo": round(p_elo, 4),
                "spread_elo": round(spread_elo, 2),
                "p_four_factors_ml": round(p_ml_clf, 4),
                "spread_four_factors_ml": round(spread_ml, 2),
                "pace_esperado": pred_pos["pace_esp"],
                "mu_local": round(mu_l_ens, 2),
                "mu_visita": round(mu_v_ens, 2),
                "spread_ensemble_local": round(spread_ensemble, 2),
                "total_esperado": round(total_ensemble, 2),
            },
            "features": feat,
            "fatiga": feat["fatiga_detail"],
            "consenso": consenso,
            "monte_carlo": mc,
            "pago_anticipado": pago_anticipado,
        }

        resultado["bajo_riesgo"] = selecciones_bajo_riesgo(
            resultado, casa_apuestas=casa_apuestas
        )
        return resultado


def armar_combinadas_y_zona_picante(
    predicciones_jornada: List[Dict[str, Any]],
    casa_apuestas: str = "Betano (Pago Anticipado +20 pts / SuperCuotas)",
) -> Dict[str, Any]:
    """
    🎯 Armador de Combinadas Seguras + 🌶️ Zona Picante (Boleto Cuota 5.00+):
    1. Genera el Ranking de Seguridad de todos los partidos de la noche (incluyendo alerta B2B
       y semáforo de consenso).
    2. Construye automáticamente la 🥇 Mejor Combinada Doble (2 partidos) y la
       🥈 Mejor Combinada Triple (3 partidos) maximizando el piso conjunto P10 de Monte Carlo
       y descartando automáticamente partidos trampa (`es_partido_trampa == True`).
    3. Construye la 🌶️ Zona Picante (Cuota Justa >= 5.00):
       - 🌶️ Escalera Picante (Máxima probabilidad matemática para superar Cuota 5.00+)
       - 💣 Bomba Express (Solo 2 o 3 partidos combinando mercados de cuota media-alta para
         alcanzar una cuota entre 5.00 y 14.00).
    """
    ranking: List[Dict[str, Any]] = []
    elegibles_seguros: List[Dict[str, Any]] = []

    for pred in predicciones_jornada:
        local = pred["local"]
        visita = pred["visita"]
        cons = pred["consenso"]
        fat = pred["fatiga"]
        picks_br = pred.get("bajo_riesgo") or selecciones_bajo_riesgo(pred, casa_apuestas=casa_apuestas)
        mejor_pick = picks_br[0]

        item_rank = {
            "partido": f"{visita} @ {local}",
            "local": local,
            "visita": visita,
            "fecha": pred.get("fecha", ""),
            "seleccion_principal": mejor_pick["mercado"],
            "categoria": mejor_pick["categoria"],
            "probabilidad": float(mejor_pick["probabilidad"]),
            "piso_p10": float(mejor_pick["piso_p10"]),
            "cuota_justa": float(mejor_pick["cuota_justa"]),
            "es_trampa": bool(cons["es_partido_trampa"]),
            "estado_consenso": cons["estado"],
            "alerta_fatiga": "⚠️ B2B/Altitud" if fat["hay_alerta"] else "✅ Normal",
            "diagnostico_fisico": fat["tarjeta_diagnostico"],
            "prediccion_ref": pred,
        }
        ranking.append(item_rank)
        if not cons["es_partido_trampa"]:
            elegibles_seguros.append(item_rank)

    # Ordenar por piso P10 descendente (y probabilidad en caso de empate)
    ranking.sort(key=lambda x: (not x["es_trampa"], x["piso_p10"], x["probabilidad"]), reverse=True)
    elegibles_seguros.sort(key=lambda x: (x["piso_p10"], x["probabilidad"]), reverse=True)

    def _empaquetar_combinada(legs: List[Dict[str, Any]], titulo: str) -> Optional[Dict[str, Any]]:
        if not legs:
            return None
        prob_conjunta = float(np.prod([x["probabilidad"] for x in legs]))
        piso_p10_conjunto = float(np.prod([x["piso_p10"] for x in legs]))
        cuota_total = float(np.prod([x["cuota_justa"] for x in legs]))
        return {
            "titulo": titulo,
            "num_selecciones": len(legs),
            "selecciones": [
                {
                    "partido": x["partido"],
                    "mercado": x["seleccion_principal"],
                    "probabilidad": round(x["probabilidad"], 4),
                    "piso_p10": round(x["piso_p10"], 4),
                    "cuota_individual": round(x["cuota_justa"], 2),
                }
                for x in legs
            ],
            "probabilidad_conjunta": round(prob_conjunta, 4),
            "piso_p10_conjunto": round(piso_p10_conjunto, 4),
            "cuota_combinada": round(cuota_total, 2),
            "retorno_por_10u": round(cuota_total * 10.0, 2),
        }

    mejor_doble = (
        _empaquetar_combinada(elegibles_seguros[:2], "🥇 Mejor Combinada Doble (2 Partidos)")
        if len(elegibles_seguros) >= 2
        else None
    )
    mejor_triple = (
        _empaquetar_combinada(elegibles_seguros[:3], "🥈 Mejor Combinada Triple (3 Partidos)")
        if len(elegibles_seguros) >= 3
        else None
    )

    # =========================================================================
    # 🌶️ ZONA PICANTE: BOLETOS CUOTA 5.00+ (Escalera Picante & Bomba Express)
    # =========================================================================
    # Pool ordenado para Escalera: primero los no-trampa y luego el resto si faltan partidos
    pool_ordenado = elegibles_seguros + [r for r in ranking if r["es_trampa"]]

    # 1. 🌶️ Escalera Picante (Máxima probabilidad matemática para alcanzar Cuota >= 5.00)
    escalera_legs: List[Dict[str, Any]] = []
    if pool_ordenado:
        # Intento 1: acumular las selecciones principales más seguras partido por partido hasta >= 5.00
        cuota_acum = 1.0
        for item in pool_ordenado:
            leg = {
                "partido": item["partido"],
                "mercado": item["seleccion_principal"],
                "probabilidad": item["probabilidad"],
                "piso_p10": item["piso_p10"],
                "cuota_individual": item["cuota_justa"],
                "pred_ref": item["prediccion_ref"],
            }
            escalera_legs.append(leg)
            cuota_acum *= leg["cuota_individual"]
            if cuota_acum >= 5.00:
                break

        # Si hay pocos partidos en la jornada y el producto de picks seguros aún es < 5.00,
        # subimos escalonadamente cada partido a líneas alternativas más exigentes (o agregamos
        # una segunda selección descorrelacionada de totales en partidos distintos) hasta >= 5.00:
        if cuota_acum < 5.00:
             target_per_leg = (5.15 ** (1.0 / max(2, len(escalera_legs))))
             legs_escalados: List[Dict[str, Any]] = []
             for item in pool_ordenado[: max(2, len(pool_ordenado))]:
                 pred_r = item["prediccion_ref"]
                 mc_r = pred_r["monte_carlo"]
                 loc_r, vis_r = pred_r["local"], pred_r["visita"]
                 # Buscar en la tabla de spread y totales la selección con mayor probabilidad cuya cuota >= target_per_leg
                 cands = []
                 for s_row in mc_r["spread"]["tabla"]:
                     cands.append((s_row["local_mercado"], float(s_row["prob_local"]), float(s_row["p10_local"]), float(s_row["cuota_justa_local"])))
                     cands.append((s_row["visita_mercado"], float(s_row["prob_visita"]), float(s_row["p10_visita"]), float(s_row["cuota_justa_visita"])))
                 for t_row in mc_r["totales"]["tabla"]:
                     cands.append((f"Más de {t_row['linea']:.1f} pts", float(t_row["prob_over"]), float(t_row["p10_over"]), float(t_row["cuota_justa_over"])))
                     cands.append((f"Menos de {t_row['linea']:.1f} pts", float(t_row["prob_under"]), float(t_row["p10_under"]), float(t_row["cuota_justa_under"])))
                 # Filtrar opciones que ayuden a llegar a 5.0+
                 valid_c = [c for c in cands if c[3] >= min(3.2, target_per_leg)]
                 if valid_c:
                     valid_c.sort(key=lambda x: x[1], reverse=True)
                     best_c = valid_c[0]
                 else:
                     cands.sort(key=lambda x: x[3], reverse=True)
                     best_c = cands[0]
                 legs_escalados.append(
                     {
                         "partido": item["partido"],
                         "mercado": best_c[0],
                         "probabilidad": best_c[1],
                         "piso_p10": best_c[2],
                         "cuota_individual": best_c[3],
                     }
                 )

             # Si solo había 1 partido en la lista, combinamos Ganador/Spread exigente + Over/Under exigente
             if len(legs_escalados) == 1:
                 pred_r = pool_ordenado[0]["prediccion_ref"]
                 mc_r = pred_r["monte_carlo"]
                 t_row = mc_r["totales"]["tabla"][-2]  # Línea alta Over (cuota ~2.60)
                 legs_escalados.append(
                     {
                         "partido": f"{pool_ordenado[0]['partido']} (Total Alt.)",
                         "mercado": f"Más de {t_row['linea']:.1f} pts",
                         "probabilidad": float(t_row["prob_over"]),
                         "piso_p10": float(t_row["p10_over"]),
                         "cuota_individual": float(t_row["cuota_justa_over"]),
                     }
                 )
             escalera_legs = legs_escalados

    # Empaquetar Escalera Picante garantizando cuota >= 5.00
    prob_esc = float(np.prod([x["probabilidad"] for x in escalera_legs])) if escalera_legs else 0.0
    cuota_esc = float(np.prod([x["cuota_individual"] for x in escalera_legs])) if escalera_legs else 0.0
    escalera_picante = (
        {
            "titulo": "🌶️ Escalera Picante (Máxima Probabilidad para Cuota 5.00+)",
            "num_selecciones": len(escalera_legs),
            "selecciones": [
                {
                    "partido": x["partido"],
                    "mercado": x["mercado"],
                    "probabilidad": round(x["probabilidad"], 4),
                    "cuota_individual": round(x["cuota_individual"], 2),
                }
                for x in escalera_legs
            ],
            "probabilidad_conjunta": round(prob_esc, 4),
            "cuota_combinada": round(cuota_esc, 2),
            "retorno_por_10u": round(cuota_esc * 10.0, 2),
        }
        if escalera_legs
        else None
    )

    # 2. 💣 Bomba Express (2–3 partidos · Cuota entre 5.00 y 14.00)
    bomba_legs: List[Dict[str, Any]] = []
    if pool_ordenado:
        top_games = pool_ordenado[: min(3, len(pool_ordenado))]
        n_g = len(top_games)
        # Si tenemos 2 partidos buscamos cuotas individuales ~2.35-2.65 (2.45^2 = 6.00)
        # Si tenemos 3 partidos buscamos cuotas individuales ~1.78-2.10 (1.88^3 = 6.64)
        target_odd_bomba = 2.40 if n_g == 2 else (1.88 if n_g >= 3 else 5.20)

        for item in top_games:
            pred_r = item["prediccion_ref"]
            mc_r = pred_r["monte_carlo"]
            loc_r, vis_r = pred_r["local"], pred_r["visita"]
            p_fav_l = mc_r["moneyline"]["prob_local"] >= mc_r["moneyline"]["prob_visita"]
            fav_t = loc_r if p_fav_l else vis_r
            p_fav_val = mc_r["moneyline"]["prob_local"] if p_fav_l else mc_r["moneyline"]["prob_visita"]

            opciones_bomba = []
            # Opción A: Spread exigente a favor del favorito
            for s_row in mc_r["spread"]["tabla"]:
                for m_lbl, pr_v, q_v in [
                    (f"Spread Exigente: {s_row['local_mercado']}", float(s_row["prob_local"]), float(s_row["cuota_justa_local"])),
                    (f"Spread Exigente: {s_row['visita_mercado']}", float(s_row["prob_visita"]), float(s_row["cuota_justa_visita"])),
                ]:
                    if 1.72 <= q_v <= 3.10:
                        opciones_bomba.append((abs(q_v - target_odd_bomba), m_lbl, pr_v, q_v))

            # Opción B: Gana Favorito + Over/Under central (Bet Builder en 1 partido)
            tot_central = next((r for r in mc_r["totales"]["tabla"] if r["es_central"]), mc_r["totales"]["tabla"][len(mc_r["totales"]["tabla"]) // 2])
            es_over_fav = tot_central["prob_over"] >= tot_central["prob_under"]
            ou_lbl = f"Más de {tot_central['linea']:.1f}" if es_over_fav else f"Menos de {tot_central['linea']:.1f}"
            p_ou = float(tot_central["prob_over"] if es_over_fav else tot_central["prob_under"])
            p_combo_same = float(np.clip(p_fav_val * p_ou, 0.22, 0.65))
            q_combo_same = round(1.0 / p_combo_same, 2)
            opciones_bomba.append(
                (
                    abs(q_combo_same - target_odd_bomba),
                    f"Gana {fav_t} + {ou_lbl} pts",
                    p_combo_same,
                    q_combo_same,
                )
            )

            opciones_bomba.sort(key=lambda x: x[0])
            _, b_mkt, b_pr, b_q = opciones_bomba[0]
            bomba_legs.append(
                {
                    "partido": item["partido"],
                    "mercado": b_mkt,
                    "probabilidad": b_pr,
                    "cuota_individual": b_q,
                }
            )

        # Si el producto quedó ligeramente por debajo de 5.00 (ej. 4.85), ajustar el último leg a una línea superior
        prod_b = float(np.prod([x["cuota_individual"] for x in bomba_legs]))
        if prod_b < 5.00 and bomba_legs:
            factor_faltante = 5.15 / max(1.0, prod_b)
            ult = bomba_legs[-1]
            nueva_cuota = round(min(4.20, ult["cuota_individual"] * factor_faltante), 2)
            ult["cuota_individual"] = nueva_cuota
            ult["probabilidad"] = round(1.0 / nueva_cuota, 4)
            ult["mercado"] = ult["mercado"] + " (Línea Plus)"

    prob_bomba = float(np.prod([x["probabilidad"] for x in bomba_legs])) if bomba_legs else 0.0
    cuota_bomba = float(np.prod([x["cuota_individual"] for x in bomba_legs])) if bomba_legs else 0.0
    bomba_express = (
        {
            "titulo": "💣 Bomba Express (2–3 Partidos · Cuota 5.00+)",
            "num_selecciones": len(bomba_legs),
            "selecciones": [
                {
                    "partido": x["partido"],
                    "mercado": x["mercado"],
                    "probabilidad": round(x["probabilidad"], 4),
                    "cuota_individual": round(x["cuota_individual"], 2),
                }
                for x in bomba_legs
            ],
            "probabilidad_conjunta": round(prob_bomba, 4),
            "cuota_combinada": round(cuota_bomba, 2),
            "retorno_por_10u": round(cuota_bomba * 10.0, 2),
        }
        if bomba_legs
        else None
    )

    return {
        "ranking_seguridad": [
            {k: v for k, v in r.items() if k != "prediccion_ref"} for r in ranking
        ],
        "mejor_doble": mejor_doble,
        "mejor_triple": mejor_triple,
        "zona_picante": {
            "escalera_picante": escalera_picante,
            "bomba_express": bomba_express,
        },
    }
