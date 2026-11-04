"""
Motor Cronológico de Features Avanzadas, Four Factors (EWMA Híbrido 70/30)
y Factores Tácticos Exclusivos NBA (`engine/features.py`):
- Cero Fuga de Información (Zero Data Leakage): cada feature para un partido en fecha `t`
  se calcula exclusivamente con partidos estrictamente anteriores a `t`.
- EWMA Híbrido: 70% rendimiento específico (Local cuando juega en casa / Visita cuando juega fuera)
  + 30% rendimiento General.
- Four Factors de Dean Oliver:
  1. eFG% = (FGM + 0.5 * 3PM) / FGA
  2. TOV% = TOV / (FGA + 0.44 * FTA + TOV)
  3. ORB% = OREB / (OREB + Opp_DREB)
  4. FTR  = FTA / FGA
- Detector Automático de Fatiga por Calendario (`ajuste_fatiga_nba`):
  * B2B Visitante: -2.6 pts (y reduce eficiencia defensiva)
  * B2B Local: -1.5 pts
  * Densidad 3-in-4 o 4-in-6 noches: -1.2 pts adicionales
  * Rest Advantage (>= 3 días vs 1 día): +1.0 pt
- Factor Altitud de Pabellones NBA (`calcular_ajuste_altitud`):
  * Denver Nuggets (1,609 msnm): +1.6 pts base local (+40% si visitante llega en B2B o 3-in-4 -> +2.24 pts)
  * Utah Jazz (1,288 msnm): +1.1 pts base local (+40% si visitante llega en B2B o 3-in-4 -> +1.54 pts)
  * Se neutraliza (0.0 pts) cuando juegan Denver vs Utah entre sí.
"""

from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from engine.config import (
    NBA_TEAMS,
    NBA_AVG_PACE,
    NBA_AVG_ORTG,
    ALTITUD_NBA,
    FATIGA_B2B_VISITA_PTS,
    FATIGA_B2B_LOCAL_PTS,
    FATIGA_DENSIDAD_PTS,
    BONO_DESCANSO_PTS,
    PROP_Q1_BASE,
    PROP_1H_BASE,
    normalizar_equipo,
)
from engine.elo_system import EloSystemNBA


def _parse_date(d_val: Union[str, date, datetime]) -> date:
    if isinstance(d_val, datetime):
        return d_val.date()
    if isinstance(d_val, date):
        return d_val
    return datetime.strptime(str(d_val)[:10], "%Y-%m-%d").date()


def calcular_four_factors_partido(
    fgm: float,
    fga: float,
    tpm: float,
    ftm: float,
    fta: float,
    oreb: float,
    reb: float,
    tov: float,
    opp_oreb: float,
    opp_reb: float,
) -> Dict[str, float]:
    """
    Calcula los Four Factors de Dean Oliver para un equipo en un partido:
    1. eFG%: (FGM + 0.5 * 3PM) / FGA
    2. TOV%: TOV / (FGA + 0.44 * FTA + TOV)
    3. ORB%: OREB / (OREB + Opp_DREB)
    4. FTR:  FTA / FGA
    """
    fga_safe = max(1.0, float(fga))
    efg = (float(fgm) + 0.5 * float(tpm)) / fga_safe

    plays = fga_safe + 0.44 * float(fta) + float(tov)
    tov_pct = float(tov) / max(1.0, plays)

    opp_dreb = max(1.0, float(opp_reb) - float(opp_oreb))
    orb_pct = float(oreb) / max(1.0, float(oreb) + opp_dreb)

    ftr = float(fta) / fga_safe

    return {
        "efg": float(np.clip(efg, 0.35, 0.75)),
        "tov_pct": float(np.clip(tov_pct, 0.05, 0.28)),
        "orb_pct": float(np.clip(orb_pct, 0.10, 0.45)),
        "ftr": float(np.clip(ftr, 0.08, 0.55)),
    }


def analizar_calendario_equipo(
    fechas_previas: List[date], fecha_partido: date
) -> Dict[str, Any]:
    """
    Analiza las fechas jugadas por un equipo estrictamente antes de `fecha_partido`
    y determina:
      - `dias_descanso`: días de descanso completos entre el último partido y `fecha_partido`
        (0 = jugó ayer / Back-to-Back, 1 = 1 día de descanso, etc.)
      - `es_b2b`: True si jugó el día anterior (`(fecha_partido - ultima_fecha).days == 1`)
      - `es_3_en_4`: True si con el partido de hoy suma >= 3 partidos en una ventana de 4 noches
      - `es_4_en_6`: True si con el partido de hoy suma >= 4 partidos en una ventana de 6 noches
    """
    fechas_ordenadas = sorted([d for d in fechas_previas if d < fecha_partido])
    if not fechas_ordenadas:
        return {
            "dias_descanso": 3,
            "es_b2b": False,
            "es_3_en_4": False,
            "es_4_en_6": False,
            "partidos_ult_4_noches": 1,
            "partidos_ult_6_noches": 1,
        }

    ultima = fechas_ordenadas[-1]
    diff_days = (fecha_partido - ultima).days
    dias_descanso = max(0, diff_days - 1)
    es_b2b = dias_descanso == 0

    ult_3_dias = sum(1 for d in fechas_ordenadas if 1 <= (fecha_partido - d).days <= 3)
    ult_5_dias = sum(1 for d in fechas_ordenadas if 1 <= (fecha_partido - d).days <= 5)

    es_3_en_4 = (ult_3_dias + 1) >= 3
    es_4_en_6 = (ult_5_dias + 1) >= 4

    return {
        "dias_descanso": int(dias_descanso),
        "es_b2b": bool(es_b2b),
        "es_3_en_4": bool(es_3_en_4),
        "es_4_en_6": bool(es_4_en_6),
        "partidos_ult_4_noches": int(ult_3_dias + 1),
        "partidos_ult_6_noches": int(ult_5_dias + 1),
    }


def calcular_ajuste_altitud(
    local: str,
    visita: str,
    visita_en_fatiga: bool = False,
) -> Dict[str, Any]:
    """
    Calcula el Factor Altitud de Pabellones NBA (`ALTITUD_NBA`):
    - Denver Nuggets (1,609 msnm - Ball Arena): +1.6 pts local
    - Utah Jazz (1,288 msnm - Delta Center): +1.1 pts local
    - Se intensifica un +40% adicional si el visitante llega en Back-to-Back o 3-in-4.
    - Si juegan Denver vs Utah entre sí, el efecto de altitud se neutraliza (0.0 pts).
    """
    canon_l = normalizar_equipo(local)
    canon_v = normalizar_equipo(visita)

    if canon_l not in ALTITUD_NBA:
        return {
            "aplica": False,
            "bono_local_pts": 0.0,
            "altitud_msnm": NBA_TEAMS.get(canon_l, {}).get("altitud", 100),
            "arena": NBA_TEAMS.get(canon_l, {}).get("arena", "NBA Arena"),
            "neutralizado": False,
            "intensificado": False,
            "detalle": "Pabellón a nivel estándar (sin bono fisiológico de altitud).",
        }

    info_alt = ALTITUD_NBA[canon_l]
    # Neutralización mutua si juegan Denver vs Utah
    if canon_v in ALTITUD_NBA:
        return {
            "aplica": False,
            "bono_local_pts": 0.0,
            "altitud_msnm": info_alt["altitud_msnm"],
            "arena": info_alt["arena"],
            "neutralizado": True,
            "intensificado": False,
            "detalle": (
                f"⛰️ Neutralizado: {canon_l} ({info_alt['altitud_msnm']} msnm) vs "
                f"{canon_v} (ambos equipos adaptados a gran altitud → 0.0 pts)."
            ),
        }

    bono = float(info_alt["bono_pts"])
    intensificado = False
    if visita_en_fatiga:
        bono = round(bono * float(info_alt["intensificador_fatiga"]), 2)
        intensificado = True

    extra_txt = " (intensificado +40% por fatiga visitante)" if intensificado else ""
    return {
        "aplica": True,
        "bono_local_pts": float(bono),
        "altitud_msnm": info_alt["altitud_msnm"],
        "arena": info_alt["arena"],
        "neutralizado": False,
        "intensificado": intensificado,
        "detalle": (
            f"⛰️ Factor Altitud ({info_alt['arena']} · {info_alt['altitud_msnm']} msnm): "
            f"+{bono:.2f} pts a favor de {canon_l}{extra_txt}."
        ),
    }


def ajuste_fatiga_nba(
    local: str,
    visita: str,
    fechas_local: List[Union[str, date, datetime]],
    fechas_visita: List[Union[str, date, datetime]],
    fecha_partido: Union[str, date, datetime],
) -> Dict[str, Any]:
    """
    Detector Automático de Fatiga por Calendario y Altitud (`ajuste_fatiga_nba`):
    1. Back-to-Back (0 días de descanso / jugó anoche):
       - Visitante en B2B: -2.6 puntos (y penaliza eficiencia defensiva)
       - Local en B2B: -1.5 puntos
    2. Densidad 3-in-4 o 4-in-6 noches:
       - Penalización adicional de -1.2 puntos por fatiga acumulada de piernas.
    3. Ventaja de descanso (Rest Advantage):
       - Si un equipo descansó >= 3 días y el rival solo 1 día: bono de +1.0 punto.
    4. Factor Altitud NBA (Denver 1,609m / Utah 1,288m):
       - +1.6 pts en Denver / +1.1 pts en Utah (+40% si visitante viene en B2B o 3-in-4).
    """
    canon_l = normalizar_equipo(local)
    canon_v = normalizar_equipo(visita)
    f_game = _parse_date(fecha_partido)
    fl_dates = [_parse_date(d) for d in fechas_local]
    fv_dates = [_parse_date(d) for d in fechas_visita]

    cal_l = analizar_calendario_equipo(fl_dates, f_game)
    cal_v = analizar_calendario_equipo(fv_dates, f_game)

    ajuste_l = 0.0
    ajuste_v = 0.0
    notas: List[str] = []

    # 1. Back-to-Back
    if cal_l["es_b2b"]:
        ajuste_l += FATIGA_B2B_LOCAL_PTS  # -1.5 pts
        notas.append(f"{canon_l} llega en Back-to-Back de local (0 días de descanso: {FATIGA_B2B_LOCAL_PTS:+.1f} pts)")
    if cal_v["es_b2b"]:
        ajuste_v += FATIGA_B2B_VISITA_PTS  # -2.6 pts
        notas.append(f"{canon_v} llega en Back-to-Back de visitante (0 días de descanso: {FATIGA_B2B_VISITA_PTS:+.1f} pts)")

    # 2. Densidad 3-in-4 o 4-in-6 noches
    if cal_l["es_3_en_4"] or cal_l["es_4_en_6"]:
        ajuste_l += FATIGA_DENSIDAD_PTS  # -1.2 pts
        etiqueta = "3-in-4" if cal_l["es_3_en_4"] else "4-in-6"
        notas.append(f"{canon_l} sufre densidad de calendario ({etiqueta} noches: {FATIGA_DENSIDAD_PTS:+.1f} pts)")

    if cal_v["es_3_en_4"] or cal_v["es_4_en_6"]:
        ajuste_v += FATIGA_DENSIDAD_PTS  # -1.2 pts
        etiqueta = "3-in-4" if cal_v["es_3_en_4"] else "4-in-6"
        notas.append(f"{canon_v} sufre densidad de calendario ({etiqueta} noches: {FATIGA_DENSIDAD_PTS:+.1f} pts)")

    # 3. Ventaja de descanso (Rest Advantage: >= 3 días vs 1 día)
    if cal_l["dias_descanso"] >= 3 and cal_v["dias_descanso"] == 1:
        ajuste_l += BONO_DESCANSO_PTS  # +1.0 pt
        notas.append(
            f"Ventaja de descanso para {canon_l} ({cal_l['dias_descanso']} días vs 1 día de {canon_v}: +{BONO_DESCANSO_PTS:.1f} pts)"
        )
    elif cal_v["dias_descanso"] >= 3 and cal_l["dias_descanso"] == 1:
        ajuste_v += BONO_DESCANSO_PTS  # +1.0 pt
        notas.append(
            f"Ventaja de descanso para {canon_v} ({cal_v['dias_descanso']} días vs 1 día de {canon_l}: +{BONO_DESCANSO_PTS:.1f} pts)"
        )

    # 4. Factor Altitud NBA
    visita_fatigada = bool(cal_v["es_b2b"] or cal_v["es_3_en_4"] or cal_v["es_4_en_6"])
    info_altitud = calcular_ajuste_altitud(canon_l, canon_v, visita_en_fatiga=visita_fatigada)
    bono_altitud = float(info_altitud["bono_local_pts"])
    if info_altitud["aplica"] or info_altitud["neutralizado"]:
        notas.append(info_altitud["detalle"])

    # Diferencial neto a favor del local: (ajuste_l + bono_altitud) - ajuste_v
    neto_local_pts = round((ajuste_l + bono_altitud) - ajuste_v, 2)

    # Construir mensaje ejecutivo claro para la tarjeta informativa de la UI
    if notas:
        resumen_tarjeta = (
            f"✈️ Alerta de Fatiga / Altitud: {' | '.join(notas)} "
            f"→ Impacto neto en Spread Local: {neto_local_pts:+.2f} pts "
            f"(Descanso: {canon_l} {cal_l['dias_descanso']}d vs {canon_v} {cal_v['dias_descanso']}d)."
        )
    else:
        resumen_tarjeta = (
            f"✅ Calendario Equilibrado: {canon_l} ({cal_l['dias_descanso']} días de descanso) vs "
            f"{canon_v} ({cal_v['dias_descanso']} días de descanso) → Sin penalizaciones por B2B ni altitud (0.00 pts)."
        )

    return {
        "local": canon_l,
        "visita": canon_v,
        "dias_descanso_l": cal_l["dias_descanso"],
        "dias_descanso_v": cal_v["dias_descanso"],
        "b2b_l": cal_l["es_b2b"],
        "b2b_v": cal_v["es_b2b"],
        "densidad_l": cal_l["es_3_en_4"] or cal_l["es_4_en_6"],
        "densidad_v": cal_v["es_3_en_4"] or cal_v["es_4_en_6"],
        "ajuste_fatiga_l_pts": round(ajuste_l, 2),
        "ajuste_fatiga_v_pts": round(ajuste_v, 2),
        "bono_altitud_l_pts": round(bono_altitud, 2),
        "impacto_neto_local_pts": neto_local_pts,
        "altitud_info": info_altitud,
        "hay_alerta": bool(
            cal_l["es_b2b"]
            or cal_v["es_b2b"]
            or cal_l["es_3_en_4"]
            or cal_v["es_3_en_4"]
            or info_altitud["aplica"]
        ),
        "notas": notas,
        "tarjeta_diagnostico": resumen_tarjeta,
    }


class TeamStateTracker:
    """
    Rastreador incremental sin fuga de información (Zero Data Leakage) del estado estadístico
    de las 30 franquicias NBA:
    - Mantiene medias móviles exponenciales (EWMA) generales y específicas Local/Visita
      para combinar 70% específico (Casa/Fuera) + 30% General.
    - Registra Four Factors de Dean Oliver, Pace, ORtg, DRtg, proporciones Q1/1H,
      fechas de partidos para cálculo de fatiga y rachas de últimos 5 y 10 encuentros.
    """

    def __init__(self, alpha_ewma: float = 0.14) -> None:
        self.alpha = float(alpha_ewma)
        self.teams: Dict[str, Dict[str, Any]] = {}
        self.current_season: Optional[int] = None
        for t in NBA_TEAMS:
            self._init_team(t)

    def _init_team(self, team: str) -> None:
        meta = NBA_TEAMS.get(team, {})
        q1_def = PROP_Q1_BASE * float(meta.get("q1_mult", 1.0))
        h1_def = PROP_1H_BASE * float(meta.get("h1_mult", 1.0))
        base_metrics = {
            "ortg": NBA_AVG_ORTG,
            "drtg": NBA_AVG_ORTG,
            "pace": NBA_AVG_PACE,
            "efg": 0.542,
            "tov_pct": 0.125,
            "orb_pct": 0.242,
            "ftr": 0.255,
            "q1_ratio": q1_def,
            "h1_ratio": h1_def,
        }
        self.teams[team] = {
            "gen": dict(base_metrics),
            "home": dict(base_metrics),
            "away": dict(base_metrics),
            "pj_total": 0,
            "pj_season": 0,
            "wins_season": 0,
            "losses_season": 0,
            "fechas": [],
            "ult_resultados": [],  # 1 victoria, 0 derrota
            "ult_margenes": [],    # pts_favor - pts_contra
        }

    def _ensure_team(self, team: str) -> str:
        canon = normalizar_equipo(team)
        if canon not in self.teams:
            self._init_team(canon)
        return canon

    def iniciar_nueva_temporada(self, season: int) -> None:
        """
        Al cambiar de temporada, reinicia el conteo de PJ/Récord de la temporada actual
        y regresa suavemente las métricas EWMA un 30% hacia la media NBA.
        """
        if self.current_season is not None and int(season) != self.current_season:
            for t, st in self.teams.items():
                if st["pj_season"] > 0:
                    st["pj_prev_season"] = st["pj_season"]
                st["pj_season"] = 0
                st["wins_season"] = 0
                st["losses_season"] = 0
                for bucket in ("gen", "home", "away"):
                    st[bucket]["ortg"] = 0.70 * st[bucket]["ortg"] + 0.30 * NBA_AVG_ORTG
                    st[bucket]["drtg"] = 0.70 * st[bucket]["drtg"] + 0.30 * NBA_AVG_ORTG
                    st[bucket]["pace"] = 0.70 * st[bucket]["pace"] + 0.30 * NBA_AVG_PACE
        self.current_season = int(season)

    def obtener_perfil_hibrido(self, team: str, es_local: bool) -> Dict[str, Any]:
        """
        Retorna el perfil EWMA híbrido (70% rendimiento específico Local/Visita + 30% General)
        estrictamente con la información acumulada antes del partido.
        """
        canon = self._ensure_team(team)
        st = self.teams[canon]
        spec = st["home"] if es_local else st["away"]
        gen = st["gen"]

        out: Dict[str, Any] = {}
        for k in ("ortg", "drtg", "pace", "efg", "tov_pct", "orb_pct", "ftr", "q1_ratio", "h1_ratio"):
            out[k] = 0.70 * float(spec[k]) + 0.30 * float(gen[k])

        out["net_rtg"] = out["ortg"] - out["drtg"]
        # Si la temporada nueva recién empieza (0 PJ) pero hay base histórica sólida (>= 10 PJ totales),
        # usar la muestra efectiva ponderada para no disparar falsas alertas de "0 PJ < 8 PJ":
        pj_s = int(st["pj_season"])
        if pj_s == 0 and int(st["pj_total"]) >= 10:
            pj_s = int(st.get("pj_prev_season") or st["pj_total"])
        out["pj_season"] = pj_s
        out["pj_total"] = int(st["pj_total"])
        out["wins_season"] = int(st["wins_season"])
        out["losses_season"] = int(st["losses_season"])

        ult_res = st["ult_resultados"]
        ult_mar = st["ult_margenes"]
        out["win_pct_5"] = float(np.mean(ult_res[-5:])) if ult_res else 0.50
        out["win_pct_10"] = float(np.mean(ult_res[-10:])) if ult_res else 0.50
        out["mov_5"] = float(np.mean(ult_mar[-5:])) if ult_mar else 0.0
        out["mov_10"] = float(np.mean(ult_mar[-10:])) if ult_mar else 0.0
        out["racha_str"] = "".join("G" if r == 1 else "P" for r in ult_res[-5:]) or "—"
        out["fechas"] = list(st["fechas"])
        return out

    def registrar_partido(self, row: Union[pd.Series, Dict[str, Any]]) -> None:
        """
        Actualiza las métricas EWMA tras haberse jugado un partido.
        """
        season = row.get("temporada")
        if season is not None and pd.notna(season):
            self.iniciar_nueva_temporada(int(season))

        canon_l = self._ensure_team(str(row["local"]))
        canon_v = self._ensure_team(str(row["visita"]))
        fecha_dt = _parse_date(row["fecha"])

        pts_l = float(row["pts_l"])
        pts_v = float(row["pts_v"])
        poss = float(row.get("poss") or NBA_AVG_PACE)
        ortg_l = float(row.get("ortg_l") or ((pts_l / poss) * 100.0))
        drtg_l = float(row.get("drtg_l") or ((pts_v / poss) * 100.0))
        ortg_v = float(row.get("ortg_v") or drtg_l)
        drtg_v = float(row.get("drtg_v") or ortg_l)

        ff_l = calcular_four_factors_partido(
            row.get("fgm_l", 40), row.get("fga_l", 88), row.get("tpm_l", 12),
            row.get("ftm_l", 18), row.get("fta_l", 22), row.get("oreb_l", 10),
            row.get("reb_l", 44), row.get("tov_l", 13), row.get("oreb_v", 10), row.get("reb_v", 44)
        )
        ff_v = calcular_four_factors_partido(
            row.get("fgm_v", 40), row.get("fga_v", 88), row.get("tpm_v", 12),
            row.get("ftm_v", 18), row.get("fta_v", 22), row.get("oreb_v", 10),
            row.get("reb_v", 44), row.get("tov_v", 13), row.get("oreb_l", 10), row.get("reb_l", 44)
        )

        fue_ot = int(row.get("fue_ot") or 0)
        factor_48 = (48.0 / 53.0) if fue_ot == 1 else 1.0
        pts_l_48 = max(70.0, pts_l * factor_48)
        pts_v_48 = max(70.0, pts_v * factor_48)

        q1_l_ratio = float(np.clip(float(row.get("q1_l") or (pts_l_48 * PROP_Q1_BASE)) / pts_l_48, 0.18, 0.34))
        h1_l_ratio = float(np.clip(float(row.get("h1_l") or (pts_l_48 * PROP_1H_BASE)) / pts_l_48, 0.40, 0.62))
        q1_v_ratio = float(np.clip(float(row.get("q1_v") or (pts_v_48 * PROP_Q1_BASE)) / pts_v_48, 0.18, 0.34))
        h1_v_ratio = float(np.clip(float(row.get("h1_v") or (pts_v_48 * PROP_1H_BASE)) / pts_v_48, 0.40, 0.62))

        obs_l = {
            "ortg": ortg_l, "drtg": drtg_l, "pace": poss,
            "efg": ff_l["efg"], "tov_pct": ff_l["tov_pct"],
            "orb_pct": ff_l["orb_pct"], "ftr": ff_l["ftr"],
            "q1_ratio": q1_l_ratio, "h1_ratio": h1_l_ratio,
        }
        obs_v = {
            "ortg": ortg_v, "drtg": drtg_v, "pace": poss,
            "efg": ff_v["efg"], "tov_pct": ff_v["tov_pct"],
            "orb_pct": ff_v["orb_pct"], "ftr": ff_v["ftr"],
            "q1_ratio": q1_v_ratio, "h1_ratio": h1_v_ratio,
        }

        a = self.alpha
        for team_key, obs, spec_bucket, won, margin in [
            (canon_l, obs_l, "home", 1 if pts_l > pts_v else 0, pts_l - pts_v),
            (canon_v, obs_v, "away", 1 if pts_v > pts_l else 0, pts_v - pts_l),
        ]:
            st = self.teams[team_key]
            for k, val in obs.items():
                st["gen"][k] = (1.0 - a) * st["gen"][k] + a * float(val)
                st[spec_bucket][k] = (1.0 - a) * st[spec_bucket][k] + a * float(val)
            st["pj_total"] += 1
            st["pj_season"] += 1
            if won == 1:
                st["wins_season"] += 1
            else:
                st["losses_season"] += 1
            st["fechas"].append(fecha_dt)
            st["ult_resultados"].append(won)
            st["ult_margenes"].append(float(margin))


def extraer_features_pre_partido(
    local: str,
    visita: str,
    fecha: Union[str, date, datetime],
    tracker: TeamStateTracker,
    elo_sys: EloSystemNBA,
) -> Dict[str, Any]:
    """
    Extrae el vector completo de features pre-partido (Elo, EWMA Híbrido 70/30, Four Factors,
    racha 5 y 10 partidos, fatiga B2B/densidad y altitud) sin usar datos del partido actual.
    """
    canon_l = normalizar_equipo(local)
    canon_v = normalizar_equipo(visita)

    perf_l = tracker.obtener_perfil_hibrido(canon_l, es_local=True)
    perf_v = tracker.obtener_perfil_hibrido(canon_v, es_local=False)

    fatiga = ajuste_fatiga_nba(
        canon_l, canon_v, perf_l["fechas"], perf_v["fechas"], fecha
    )
    elo_pred = elo_sys.predecir_partido(
        canon_l, canon_v, ajuste_pts_extra=fatiga["impacto_neto_local_pts"]
    )

    return {
        "local": canon_l,
        "visita": canon_v,
        "fecha": str(fecha)[:10],
        # Elo
        "elo_l": elo_pred["elo_local"],
        "elo_v": elo_pred["elo_visita"],
        "elo_diff": elo_pred["diff_elo"],
        "p_elo": elo_pred["p_elo"],
        "spread_elo": elo_pred["spread_elo"],
        # Eficiencia EWMA Híbrida (70% específico + 30% general)
        "ortg_l": perf_l["ortg"],
        "drtg_l": perf_l["drtg"],
        "net_rtg_l": perf_l["net_rtg"],
        "pace_l": perf_l["pace"],
        "ortg_v": perf_v["ortg"],
        "drtg_v": perf_v["drtg"],
        "net_rtg_v": perf_v["net_rtg"],
        "pace_v": perf_v["pace"],
        "net_rtg_diff": perf_l["net_rtg"] - perf_v["net_rtg"],
        # Four Factors de Dean Oliver (Diferenciales Local - Visita)
        "efg_l": perf_l["efg"],
        "efg_v": perf_v["efg"],
        "efg_diff": perf_l["efg"] - perf_v["efg"],
        "tov_pct_l": perf_l["tov_pct"],
        "tov_pct_v": perf_v["tov_pct"],
        "tov_pct_diff": perf_l["tov_pct"] - perf_v["tov_pct"],
        "orb_pct_l": perf_l["orb_pct"],
        "orb_pct_v": perf_v["orb_pct"],
        "orb_pct_diff": perf_l["orb_pct"] - perf_v["orb_pct"],
        "ftr_l": perf_l["ftr"],
        "ftr_v": perf_v["ftr"],
        "ftr_diff": perf_l["ftr"] - perf_v["ftr"],
        # Proporciones Q1 y 1H
        "q1_ratio_l": perf_l["q1_ratio"],
        "h1_ratio_l": perf_l["h1_ratio"],
        "q1_ratio_v": perf_v["q1_ratio"],
        "h1_ratio_v": perf_v["h1_ratio"],
        # Forma reciente (5 y 10 partidos)
        "win_pct_5_diff": perf_l["win_pct_5"] - perf_v["win_pct_5"],
        "win_pct_10_diff": perf_l["win_pct_10"] - perf_v["win_pct_10"],
        "mov_5_diff": perf_l["mov_5"] - perf_v["mov_5"],
        "mov_10_diff": perf_l["mov_10"] - perf_v["mov_10"],
        # Partidos jugados en la temporada
        "pj_season_l": perf_l["pj_season"],
        "pj_season_v": perf_v["pj_season"],
        "min_pj_season": min(perf_l["pj_season"], perf_v["pj_season"]),
        # Factores tácticos (Fatiga + Altitud)
        "impacto_tactico_pts": fatiga["impacto_neto_local_pts"],
        "ajuste_fatiga_l_pts": fatiga["ajuste_fatiga_l_pts"],
        "ajuste_fatiga_v_pts": fatiga["ajuste_fatiga_v_pts"],
        "bono_altitud_l_pts": fatiga["bono_altitud_l_pts"],
        "b2b_l": int(fatiga["b2b_l"]),
        "b2b_v": int(fatiga["b2b_v"]),
        "fatiga_detail": fatiga,
    }


FEATURE_COLS_ML = [
    "elo_diff",
    "net_rtg_diff",
    "efg_diff",
    "tov_pct_diff",
    "orb_pct_diff",
    "ftr_diff",
    "win_pct_5_diff",
    "mov_10_diff",
    "impacto_tactico_pts",
]


def construir_dataset_cronologico(
    df: pd.DataFrame, alpha_ewma: float = 0.14
) -> Tuple[pd.DataFrame, TeamStateTracker, EloSystemNBA]:
    """
    Recorre estrictamente en orden cronológico todos los partidos de `df`,
    calculando las features de cada partido ANTES de actualizar el estado con su resultado
    (Zero Data Leakage garantizado).
    Retorna:
      - `df_features`: DataFrame con todas las features pre-partido y targets reales
      - `tracker`: Estado actualizado al final del historial
      - `elo_sys`: Sistema Elo actualizado al final del historial
    """
    tracker = TeamStateTracker(alpha_ewma=alpha_ewma)
    elo_sys = EloSystemNBA()

    if df.empty:
        return pd.DataFrame(), tracker, elo_sys

    df_sorted = df.sort_values(["fecha", "id"]).reset_index(drop=True)
    rows_out: List[Dict[str, Any]] = []

    for _, row in df_sorted.iterrows():
        season = row.get("temporada")
        if season is not None and pd.notna(season):
            tracker.iniciar_nueva_temporada(int(season))
            if elo_sys.current_season is not None and int(season) != elo_sys.current_season:
                elo_sys._aplicar_regresion_temporada()
            elo_sys.current_season = int(season)

        feat = extraer_features_pre_partido(
            local=str(row["local"]),
            visita=str(row["visita"]),
            fecha=str(row["fecha"]),
            tracker=tracker,
            elo_sys=elo_sys,
        )
        # Adjuntar resultados reales únicamente como targets de entrenamiento / evaluación
        pts_l = int(row["pts_l"])
        pts_v = int(row["pts_v"])
        feat["id"] = row.get("id", "")
        feat["temporada"] = int(row["temporada"]) if pd.notna(row.get("temporada")) else 2026
        feat["fase"] = str(row.get("fase", "Regular"))
        feat["pts_l"] = pts_l
        feat["pts_v"] = pts_v
        feat["fue_ot"] = int(row.get("fue_ot") or 0)
        feat["target_win_l"] = 1 if pts_l > pts_v else 0
        feat["target_margin_l"] = float(pts_l - pts_v)
        feat["target_total"] = float(pts_l + pts_v)
        feat["poss_real"] = float(row.get("poss") or NBA_AVG_PACE)
        feat["cuota_l"] = row.get("cuota_l")
        feat["cuota_v"] = row.get("cuota_v")
        feat["spread_linea"] = row.get("spread_linea")
        feat["total_linea"] = row.get("total_linea")

        rows_out.append(feat)

        # Actualizar estado post-partido (después de haber extraído features pre-partido)
        tracker.registrar_partido(row)
        elo_sys.actualizar_partido(
            local=str(row["local"]),
            visita=str(row["visita"]),
            pts_l=pts_l,
            pts_v=pts_v,
            temporada=feat["temporada"],
            fecha=str(row["fecha"]),
        )

    return pd.DataFrame(rows_out), tracker, elo_sys


def generar_tabla_power_ranking(
    tracker: TeamStateTracker, elo_sys: EloSystemNBA
) -> pd.DataFrame:
    """
    Construye la tabla de Power Ranking NBA ordenada por Elo y Net Rating mostrando:
    `Pos`, `Equipo`, `Elo`, `PJ`, `Récord`, `ORtg (EWMA)`, `DRtg (EWMA)`,
    `Net Rating`, `Pace (Posesiones)` y `Racha (últ. 5)`.
    """
    filas = []
    for team in NBA_TEAMS:
        st = tracker.teams.get(team)
        if not st:
            continue
        gen = st["gen"]
        elo_val = elo_sys.get_rating(team)
        ortg_val = float(gen["ortg"])
        drtg_val = float(gen["drtg"])
        net_val = ortg_val - drtg_val
        pace_val = float(gen["pace"])
        pj = int(st["pj_season"])
        wins = int(st["wins_season"])
        losses = int(st["losses_season"])
        ult5 = st["ult_resultados"][-5:]
        racha = "".join("G" if r == 1 else "P" for r in ult5) if ult5 else "—"

        filas.append(
            {
                "Equipo": team,
                "Elo": round(elo_val, 1),
                "PJ": pj,
                "Récord": f"{wins}-{losses}",
                "ORtg (EWMA)": round(ortg_val, 1),
                "DRtg (EWMA)": round(drtg_val, 1),
                "Net Rating": round(net_val, 2),
                "Pace (Posesiones)": round(pace_val, 1),
                "Racha (últ. 5)": racha,
            }
        )

    df_rank = pd.DataFrame(filas)
    if not df_rank.empty:
        df_rank = df_rank.sort_values(
            by=["Elo", "Net Rating"], ascending=[False, False]
        ).reset_index(drop=True)
        df_rank.insert(0, "Pos", range(1, len(df_rank) + 1))
    return df_rank
