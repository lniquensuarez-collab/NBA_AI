"""
Módulo de Base de Datos SQLite para NBA ULTIMATE AI (`engine/db.py`).
- Esquema completo de partidos con marcadores finales, Q1, 1H, OT, Boxscore avanzado,
  posesiones (Dean Oliver), ORtg/DRtg normalizados a 48 min y cuotas de mercado.
- Migraciones automáticas de columnas sin pérdida de datos.
- Carga e inserción limpia sin duplicados (UPSERT por ID o por fecha+local+visita).
- Desmargenado de cuotas por el Método de Potencia (Power Method).
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from engine.config import DB_PATH, normalizar_equipo, NBA_AVG_PACE, NBA_AVG_ORTG


EXPECTED_COLUMNS: Dict[str, str] = {
    "id": "TEXT PRIMARY KEY",
    "temporada": "INTEGER",
    "fase": "TEXT DEFAULT 'Regular'",
    "fecha": "TEXT",
    "local": "TEXT",
    "visita": "TEXT",
    "pts_l": "INTEGER",
    "pts_v": "INTEGER",
    "fue_ot": "INTEGER DEFAULT 0",
    "q1_l": "INTEGER",
    "q1_v": "INTEGER",
    "h1_l": "INTEGER",
    "h1_v": "INTEGER",
    "fgm_l": "REAL",
    "fga_l": "REAL",
    "fgm_v": "REAL",
    "fga_v": "REAL",
    "tpm_l": "REAL",
    "tpa_l": "REAL",
    "tpm_v": "REAL",
    "tpa_v": "REAL",
    "ftm_l": "REAL",
    "fta_l": "REAL",
    "ftm_v": "REAL",
    "fta_v": "REAL",
    "oreb_l": "REAL",
    "reb_l": "REAL",
    "oreb_v": "REAL",
    "reb_v": "REAL",
    "ast_l": "REAL",
    "ast_v": "REAL",
    "tov_l": "REAL",
    "tov_v": "REAL",
    "poss": "REAL",
    "ortg_l": "REAL",
    "drtg_l": "REAL",
    "ortg_v": "REAL",
    "drtg_v": "REAL",
    "cuota_l": "REAL",
    "cuota_v": "REAL",
    "spread_linea": "REAL",
    "total_linea": "REAL",
}


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Crea y retorna una conexión SQLite asegurando que el directorio exista."""
    target = db_path or DB_PATH
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Inicializa la tabla `partidos` en SQLite y ejecuta migraciones automáticas
    para asegurar que todas las columnas requeridas existan.
    """
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cols_sql = ",\n    ".join(f"{col} {dtype}" for col, dtype in EXPECTED_COLUMNS.items())
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS partidos (
                {cols_sql},
                UNIQUE(fecha, local, visita)
            )
            """
        )
        # Migración automática de columnas faltantes
        cur.execute("PRAGMA table_info(partidos)")
        existing_cols = {row["name"] for row in cur.fetchall()}
        for col, dtype in EXPECTED_COLUMNS.items():
            if col not in existing_cols:
                clean_type = dtype.replace("PRIMARY KEY", "").strip()
                cur.execute(f"ALTER TABLE partidos ADD COLUMN {col} {clean_type}")

        # Índices para consultas cronológicas rápidas
        cur.execute("CREATE INDEX IF NOT EXISTS idx_partidos_fecha ON partidos(fecha)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_partidos_temporada ON partidos(temporada)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_partidos_equipos ON partidos(local, visita)")

        # Tabla SQLite de Estadísticas Reales y Estado de Lesiones de Jugadores NBA (ESPN Live)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jugadores_stats (
                id TEXT PRIMARY KEY,
                jugador TEXT,
                equipo TEXT,
                pos TEXT,
                pj INTEGER DEFAULT 0,
                min REAL,
                pts REAL,
                reb REAL,
                ast REAL,
                tpm REAL,
                pra REAL,
                pts_l5 REAL,
                reb_l5 REAL,
                ast_l5 REAL,
                tpm_l5 REAL,
                pra_l5 REAL,
                sd_pts REAL,
                sd_reb REAL,
                sd_ast REAL,
                usg REAL,
                impacto_pts REAL,
                estado_lesion TEXT DEFAULT 'Activa',
                detalle_lesion TEXT DEFAULT '',
                fuente TEXT DEFAULT 'ESPN Live',
                actualizado_en TEXT,
                UNIQUE(jugador, equipo)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jugadores_equipo ON jugadores_stats(equipo)")
        conn.commit()
    finally:
        conn.close()


def calcular_posesiones_dean_oliver(
    fga: float, tov: float, fta: float, oreb: float
) -> float:
    """
    Calcula las posesiones estimadas mediante la fórmula estándar de Dean Oliver:
    Poss = 0.96 * (FGA + TOV + 0.44 * FTA - OREB)
    """
    poss = 0.96 * (float(fga) + float(tov) + 0.44 * float(fta) - float(oreb))
    return float(max(70.0, min(130.0, poss)))


def imputar_estadisticas_boxscore(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Garantiza que cada partido tenga Q1, 1H, Boxscore avanzado, Posesiones (Dean Oliver)
    y ORtg/DRtg normalizados a 48 minutos (incluso cuando viene solo el marcador del
    endpoint de schedule sin boxscore completo).
    """
    rec = dict(record)
    pts_l = int(rec.get("pts_l") or 0)
    pts_v = int(rec.get("pts_v") or 0)
    fue_ot = int(rec.get("fue_ot") or 0)

    # Normalización de puntos a 48 minutos si hubo prórroga (53 min o más)
    # 1 OT = 53 min -> factor 48/53 = 0.90566
    factor_48 = (48.0 / 53.0) if fue_ot == 1 else 1.0
    pts_l_48 = pts_l * factor_48
    pts_v_48 = pts_v * factor_48

    # Imputación calibrada de Q1 (~25.2%) y 1H (~50.8%) si no están presentes
    if rec.get("q1_l") is None or pd.isna(rec.get("q1_l")):
        rec["q1_l"] = int(round(pts_l_48 * 0.252))
    if rec.get("q1_v") is None or pd.isna(rec.get("q1_v")):
        rec["q1_v"] = int(round(pts_v_48 * 0.252))
    if rec.get("h1_l") is None or pd.isna(rec.get("h1_l")):
        rec["h1_l"] = int(round(pts_l_48 * 0.508))
    if rec.get("h1_v") is None or pd.isna(rec.get("h1_v")):
        rec["h1_v"] = int(round(pts_v_48 * 0.508))

    # Verificar si hay estadísticas de tiro reales o si debemos imputarlas calibradas por puntaje
    has_box = (
        rec.get("fga_l") is not None
        and not pd.isna(rec.get("fga_l"))
        and float(rec.get("fga_l") or 0) > 40
    )

    if not has_box:
        # Imputación calibrada en base al puntaje en 48 minutos y ritmo histórico NBA
        total_48 = pts_l_48 + pts_v_48
        pace_est = NBA_AVG_PACE + (total_48 - 228.0) * 0.18
        pace_est = float(np.clip(pace_est, 91.0, 108.0))

        for side, pts_side in [("l", pts_l_48), ("v", pts_v_48)]:
            fga = round(pace_est * 0.885, 1)
            fta = round(max(12.0, pts_side * 0.195), 1)
            ftm = round(fta * 0.785, 1)
            pts_from_fg = max(60.0, pts_side - ftm)
            # Proporción de triples (~39% de intentos de campo)
            tpa = round(fga * 0.39, 1)
            tpm = round(min(tpa * 0.48, max(6.0, (pts_from_fg - fga * 0.72) * 0.45)), 1)
            fgm = round(max(25.0, (pts_from_fg - tpm) / 2.0), 1)
            fgm = min(fga - 5.0, fgm)
            oreb = round(max(6.0, (fga - fgm) * 0.235), 1)
            reb = round(oreb + 33.0, 1)
            ast = round(fgm * 0.625, 1)
            # Despejar TOV para que la fórmula de Dean Oliver produzca exactamente pace_est
            tov = round(max(9.0, (pace_est / 0.96) - fga - 0.44 * fta + oreb), 1)

            rec[f"fgm_{side}"] = fgm
            rec[f"fga_{side}"] = fga
            rec[f"tpm_{side}"] = tpm
            rec[f"tpa_{side}"] = tpa
            rec[f"ftm_{side}"] = ftm
            rec[f"fta_{side}"] = fta
            rec[f"oreb_{side}"] = oreb
            rec[f"reb_{side}"] = reb
            rec[f"ast_{side}"] = ast
            rec[f"tov_{side}"] = tov

    # Calcular posesiones con la fórmula de Dean Oliver (normalizando a 48 min si hubo OT)
    poss_l = calcular_posesiones_dean_oliver(
        rec["fga_l"], rec["tov_l"], rec["fta_l"], rec["oreb_l"]
    )
    poss_v = calcular_posesiones_dean_oliver(
        rec["fga_v"], rec["tov_v"], rec["fta_v"], rec["oreb_v"]
    )
    poss_raw = 0.5 * (poss_l + poss_v)
    poss_48 = poss_raw * factor_48 if has_box and fue_ot == 1 else poss_raw
    poss_48 = float(np.clip(poss_48, 84.0, 115.0))
    rec["poss"] = round(poss_48, 2)

    # Calcular ORtg y DRtg normalizados a 48 minutos (puntos por cada 100 posesiones)
    ortg_l = (pts_l_48 / poss_48) * 100.0 if poss_48 > 0 else NBA_AVG_ORTG
    ortg_v = (pts_v_48 / poss_48) * 100.0 if poss_48 > 0 else NBA_AVG_ORTG
    rec["ortg_l"] = round(float(ortg_l), 2)
    rec["drtg_l"] = round(float(ortg_v), 2)
    rec["ortg_v"] = round(float(ortg_v), 2)
    rec["drtg_v"] = round(float(ortg_l), 2)

    return rec


def guardar_partidos(
    partidos: List[Dict[str, Any]], db_path: Optional[str] = None
) -> int:
    """
    Inserta o actualiza una lista de partidos en SQLite sin generar duplicados.
    Retorna el número de registros procesados.
    """
    if not partidos:
        return 0
    init_db(db_path)
    conn = get_connection(db_path)
    cols = list(EXPECTED_COLUMNS.keys())
    placeholders = ", ".join(["?"] * len(cols))
    update_clause = ", ".join(
        [f"{c}=excluded.{c}" for c in cols if c not in ("id",)]
    )

    procesados = 0
    try:
        cur = conn.cursor()
        for raw in partidos:
            local = normalizar_equipo(raw.get("local"))
            visita = normalizar_equipo(raw.get("visita"))
            fecha = str(raw.get("fecha") or "").strip()[:10]
            if not local or not visita or not fecha:
                continue
            if raw.get("pts_l") is None or raw.get("pts_v") is None:
                continue

            rec = dict(raw)
            rec["local"] = local
            rec["visita"] = visita
            rec["fecha"] = fecha
            if not rec.get("id"):
                rec["id"] = f"{fecha}_{local}_{visita}".replace(" ", "_")
            if not rec.get("temporada"):
                year = int(fecha[:4])
                month = int(fecha[5:7])
                rec["temporada"] = year + 1 if month >= 10 else year
            if not rec.get("fase"):
                rec["fase"] = "Regular"

            rec = imputar_estadisticas_boxscore(rec)

            # Evitar conflicto UNIQUE(fecha, local, visita) si llega con otro ID
            cur.execute(
                "SELECT id FROM partidos WHERE fecha = ? AND local = ? AND visita = ?",
                (fecha, local, visita),
            )
            existing = cur.fetchone()
            if existing:
                rec["id"] = existing["id"]

            values = [rec.get(c) for c in cols]
            cur.execute(
                f"""
                INSERT INTO partidos ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT(id) DO UPDATE SET {update_clause}
                """,
                values,
            )
            procesados += 1
        conn.commit()
    finally:
        conn.close()
    return procesados


def cargar_partidos_df(
    db_path: Optional[str] = None,
    temporadas: Optional[List[int]] = None,
    fase: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carga todos los partidos de SQLite ordenados estrictamente de forma cronológica.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM partidos WHERE pts_l IS NOT NULL AND pts_v IS NOT NULL"
        params: List[Any] = []
        if temporadas:
            placeholders = ",".join(["?"] * len(temporadas))
            query += f" AND temporada IN ({placeholders})"
            params.extend(temporadas)
        if fase and fase != "Todas":
            query += " AND fase = ?"
            params.append(fase)
        query += " ORDER BY fecha ASC, id ASC"
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()


def desmargenar_cuotas_power(
    cuota_local: float, cuota_visita: float
) -> Tuple[float, float, float]:
    """
    Desmargena un par de cuotas de mercado (Moneyline o Spread/Totales de 2 vías)
    utilizando el Método de Potencia (Power Method):
        (1 / cuota_l)^k + (1 / cuota_v)^k = 1
    donde k >= 1 corrige el sesgo favorito-longshot (FLB).
    Retorna: (prob_justa_local, prob_justa_visita, overround_pct)
    """
    if cuota_local <= 1.0 or cuota_visita <= 1.0:
        raise ValueError("Las cuotas deben ser estrictamente mayores a 1.0")

    q_l = 1.0 / float(cuota_local)
    q_v = 1.0 / float(cuota_visita)
    overround = (q_l + q_v - 1.0) * 100.0

    if abs((q_l + q_v) - 1.0) < 1e-7:
        return float(q_l), float(q_v), 0.0

    def _obj(k: float) -> float:
        return (q_l ** k) + (q_v ** k) - 1.0

    try:
        k_opt = brentq(_obj, 0.05, 10.0, xtol=1e-8)
        p_l = float(q_l ** k_opt)
        p_v = float(q_v ** k_opt)
        total = p_l + p_v
        return p_l / total, p_v / total, float(overround)
    except Exception:
        # Fallback proporcional si falla la búsqueda numérica
        total = q_l + q_v
        return float(q_l / total), float(q_v / total), float(overround)
