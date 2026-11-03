"""
Sistema Elo Profesional para NBA (`engine/elo_system.py`).
Implementa el algoritmo Elo estilo FiveThirtyEight con:
- Factor de multiplicación por Margen de Victoria (MOV) con desaceleración logarítmica
  para no sobre-premiar palizas de garbage time:
    mult_mov = ((abs(pts_l - pts_v) + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_ganador)
- Conversión directa de diferencia de Elo a:
    * Probabilidad de victoria Moneyline (`p_elo`)
    * Spread esperado por Elo (`spread_elo = diff_elo / 28.0`)
- Regresión automática hacia la media (`1500.0`) del `25%` al cambiar de temporada.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from engine.config import (
    NBA_TEAMS,
    ELO_BASE,
    ELO_K_FACTOR,
    ELO_HOME_ADVANTAGE,
    ELO_SPREAD_DIVISOR,
    ELO_SEASON_REGRESSION,
    normalizar_equipo,
)


def calcular_multiplicador_mov(pts_l: int, pts_v: int, elo_diff_ganador: float) -> float:
    """
    Calcula el multiplicador de Margen de Victoria (MOV) estilo FiveThirtyEight NBA:
        mult_mov = ((abs(pts_l - pts_v) + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_ganador)
    donde `elo_diff_ganador = elo_ganador_ajustado - elo_perdedor_ajustado`.
    """
    margen = abs(int(pts_l) - int(pts_v))
    denom = max(1.5, 7.5 + 0.006 * float(elo_diff_ganador))
    mult_mov = ((margen + 3.0) ** 0.8) / denom
    return float(mult_mov)


def probabilidad_elo(elo_local: float, elo_visita: float, hca_elo: float = ELO_HOME_ADVANTAGE) -> Tuple[float, float, float]:
    """
    Calcula a partir de los ratings Elo actuales:
      1. `diff_elo` (incluyendo ventaja de localía en puntos Elo)
      2. `p_elo` (probabilidad de victoria Moneyline del local)
      3. `spread_elo = diff_elo / 28.0` (ventaja esperada en puntos del local; >0 favorece al local)
    """
    diff_elo = (float(elo_local) + float(hca_elo)) - float(elo_visita)
    p_elo = 1.0 / (1.0 + (10.0 ** (-diff_elo / 400.0)))
    spread_elo = diff_elo / ELO_SPREAD_DIVISOR
    return float(p_elo), float(spread_elo), float(diff_elo)


class EloSystemNBA:
    """
    Motor cronológico de Elo NBA sin fuga de información (Zero Data Leakage).
    """

    def __init__(
        self,
        k_factor: float = ELO_K_FACTOR,
        home_advantage: float = ELO_HOME_ADVANTAGE,
        season_regression: float = ELO_SEASON_REGRESSION,
    ) -> None:
        self.k_factor = float(k_factor)
        self.home_advantage = float(home_advantage)
        self.season_regression = float(season_regression)
        self.ratings: Dict[str, float] = {team: ELO_BASE for team in NBA_TEAMS}
        self.current_season: Optional[int] = None
        self.history: list[Dict[str, Any]] = []

    def _aplicar_regresion_temporada(self) -> None:
        """Regresa todos los equipos un 25% hacia la media (1500.0) al iniciar nueva temporada."""
        for team in list(self.ratings.keys()):
            self.ratings[team] = (
                (1.0 - self.season_regression) * self.ratings[team]
                + self.season_regression * ELO_BASE
            )

    def get_rating(self, team: str) -> float:
        canon = normalizar_equipo(team)
        if canon not in self.ratings:
            self.ratings[canon] = ELO_BASE
        return float(self.ratings[canon])

    def predecir_partido(
        self, local: str, visita: str, ajuste_pts_extra: float = 0.0
    ) -> Dict[str, float]:
        """
        Retorna la predicción Elo antes de jugarse el partido.
        Permite incorporar un `ajuste_pts_extra` (ej. fatiga o altitud convertido a escala Elo: 1 pt = 28 Elo).
        """
        elo_l = self.get_rating(local)
        elo_v = self.get_rating(visita)
        hca_efectivo = self.home_advantage + float(ajuste_pts_extra) * ELO_SPREAD_DIVISOR
        p_elo, spread_elo, diff_elo = probabilidad_elo(elo_l, elo_v, hca_elo=hca_efectivo)
        return {
            "elo_local": round(elo_l, 2),
            "elo_visita": round(elo_v, 2),
            "diff_elo": round(diff_elo, 2),
            "p_elo": float(np.clip(p_elo, 0.01, 0.99)),
            "spread_elo": round(spread_elo, 2),
        }

    def actualizar_partido(
        self,
        local: str,
        visita: str,
        pts_l: int,
        pts_v: int,
        temporada: Optional[int] = None,
        fecha: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Procesa un partido cronológicamente:
        1. Si cambió la temporada, aplica la regresión del 25% hacia 1500.
        2. Calcula la predicción pre-partido (sin fuga de información).
        3. Actualiza los ratings Elo post-partido usando el multiplicador MOV.
        """
        if temporada is not None:
            temp_int = int(temporada)
            if self.current_season is not None and temp_int != self.current_season:
                self._aplicar_regresion_temporada()
            self.current_season = temp_int

        canon_l = normalizar_equipo(local)
        canon_v = normalizar_equipo(visita)
        elo_l_pre = self.get_rating(canon_l)
        elo_v_pre = self.get_rating(canon_v)

        p_elo, spread_elo, diff_elo = probabilidad_elo(
            elo_l_pre, elo_v_pre, hca_elo=self.home_advantage
        )

        gano_local = 1.0 if int(pts_l) > int(pts_v) else (0.5 if int(pts_l) == int(pts_v) else 0.0)
        elo_l_adj = elo_l_pre + self.home_advantage
        elo_v_adj = elo_v_pre

        if gano_local >= 0.5:
            elo_diff_ganador = elo_l_adj - elo_v_adj
        else:
            elo_diff_ganador = elo_v_adj - elo_l_adj

        mult_mov = calcular_multiplicador_mov(int(pts_l), int(pts_v), elo_diff_ganador)
        delta_elo = self.k_factor * mult_mov * (gano_local - p_elo)

        self.ratings[canon_l] = elo_l_pre + delta_elo
        self.ratings[canon_v] = elo_v_pre - delta_elo

        info = {
            "fecha": fecha or "",
            "temporada": self.current_season,
            "local": canon_l,
            "visita": canon_v,
            "elo_l_pre": elo_l_pre,
            "elo_v_pre": elo_v_pre,
            "p_elo_pre": p_elo,
            "spread_elo_pre": spread_elo,
            "mult_mov": mult_mov,
            "delta_elo": delta_elo,
            "elo_l_post": self.ratings[canon_l],
            "elo_v_post": self.ratings[canon_v],
        }
        self.history.append(info)
        return info

    def procesar_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recorre un DataFrame de partidos ordenados cronológicamente y adjunta las columnas
        pre-partido `elo_l_pre`, `elo_v_pre`, `p_elo_pre`, `spread_elo_pre` con cero data leakage.
        """
        if df.empty:
            return df.copy()

        df_sorted = df.sort_values(["fecha", "id"]).reset_index(drop=True).copy()
        elo_l_list, elo_v_list, p_elo_list, spread_elo_list = [], [], [], []

        for _, row in df_sorted.iterrows():
            info = self.actualizar_partido(
                local=str(row["local"]),
                visita=str(row["visita"]),
                pts_l=int(row["pts_l"]),
                pts_v=int(row["pts_v"]),
                temporada=int(row["temporada"]) if pd.notna(row.get("temporada")) else None,
                fecha=str(row.get("fecha", "")),
            )
            elo_l_list.append(info["elo_l_pre"])
            elo_v_list.append(info["elo_v_pre"])
            p_elo_list.append(info["p_elo_pre"])
            spread_elo_list.append(info["spread_elo_pre"])

        df_sorted["elo_l_pre"] = elo_l_list
        df_sorted["elo_v_pre"] = elo_v_list
        df_sorted["p_elo_pre"] = p_elo_list
        df_sorted["spread_elo_pre"] = spread_elo_list
        return df_sorted
