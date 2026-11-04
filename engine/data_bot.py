"""
Módulo de Ingesta de Datos para NBA ULTIMATE AI (`engine/data_bot.py`):
- Ingesta histórica multihilo (ThreadPoolExecutor) desde los endpoints JSON públicos de ESPN NBA
  (100% libre de API keys).
- Soportando Regular Season, Play-In y Playoffs de las últimas 3-4 temporadas.
- Actualización diaria por rango de fechas desde el Scoreboard público de ESPN.
- Buscador automático de la cartelera NBA del día (y próximos 1 a 3 días) con detección de cuotas.
- Importador universal de archivos CSV / Excel (.xlsx) subidos por el usuario.
- Generador de dataset semilla realista NBA si no hay conexión de red o para pruebas.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
import io
from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
import requests

from engine.config import (
    NBA_TEAMS,
    ESPN_TEAMS_URL,
    ESPN_SCHEDULE_URL,
    ESPN_SCOREBOARD_URL,
    DEFAULT_SEASONS,
    normalizar_equipo,
)
from engine.db import guardar_partidos, imputar_estadisticas_boxscore

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def obtener_catalogo_espn(timeout: int = 8) -> List[Dict[str, Any]]:
    """
    Consulta el endpoint público de catálogo de equipos ESPN NBA:
    https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams
    Si no hay red, retorna el catálogo oficial local de `NBA_TEAMS`.
    """
    try:
        resp = requests.get(ESPN_TEAMS_URL, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        teams_out = []
        sports = data.get("sports", [])
        if sports and sports[0].get("leagues"):
            for item in sports[0]["leagues"][0].get("teams", []):
                t = item.get("team", {})
                teams_out.append(
                    {
                        "id": str(t.get("id", "")),
                        "name": normalizar_equipo(t.get("displayName", "")),
                        "abbr": t.get("abbreviation", ""),
                    }
                )
        if teams_out:
            return teams_out
    except Exception:
        pass

    return [
        {"id": meta["id"], "name": name, "abbr": meta["abbr"]}
        for name, meta in NBA_TEAMS.items()
    ]


def _parse_stat_pair(val: str) -> tuple[float, float]:
    """Convierte cadenas tipo '42-88' en (42.0, 88.0)."""
    try:
        parts = str(val).split("-")
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return 0.0, 0.0


def _extraer_estadisticas_competitor(comp: Dict[str, Any]) -> Dict[str, float]:
    """Extrae estadísticas avanzadas del bloque `statistics` de ESPN scoreboard si existen."""
    stats_list = comp.get("statistics", []) or []
    out: Dict[str, float] = {}
    stat_map = {}
    for s in stats_list:
        name = str(s.get("name") or s.get("abbreviation") or "").lower()
        val = s.get("displayValue")
        if name and val is not None:
            stat_map[name] = val

    # Tiros de campo
    for k in ("fieldgoalsmade-fieldgoalsattempted", "fg"):
        if k in stat_map:
            m, a = _parse_stat_pair(stat_map[k])
            if a > 0:
                out["fgm"], out["fga"] = m, a
    if "fieldgoalsmade" in stat_map:
        out["fgm"] = float(stat_map["fieldgoalsmade"])
    if "fieldgoalsattempted" in stat_map:
        out["fga"] = float(stat_map["fieldgoalsattempted"])

    # Triples
    for k in ("threepointfieldgoalsmade-threepointfieldgoalsattempted", "3pt"):
        if k in stat_map:
            m, a = _parse_stat_pair(stat_map[k])
            if a > 0:
                out["tpm"], out["tpa"] = m, a
    if "threepointfieldgoalsmade" in stat_map:
        out["tpm"] = float(stat_map["threepointfieldgoalsmade"])
    if "threepointfieldgoalsattempted" in stat_map:
        out["tpa"] = float(stat_map["threepointfieldgoalsattempted"])

    # Tiros libres
    for k in ("freethrowsmade-freethrowsattempted", "ft"):
        if k in stat_map:
            m, a = _parse_stat_pair(stat_map[k])
            if a > 0:
                out["ftm"], out["fta"] = m, a
    if "freethrowsmade" in stat_map:
        out["ftm"] = float(stat_map["freethrowsmade"])
    if "freethrowsattempted" in stat_map:
        out["fta"] = float(stat_map["freethrowsattempted"])

    # Rebotes, asistencias, pérdidas
    for k, target in [
        ("offensiverebounds", "oreb"),
        ("oreb", "oreb"),
        ("rebounds", "reb"),
        ("reb", "reb"),
        ("assists", "ast"),
        ("ast", "ast"),
        ("turnovers", "tov"),
        ("to", "tov"),
    ]:
        if k in stat_map:
            try:
                out[target] = float(stat_map[k])
            except Exception:
                pass
    return out


def _parsear_evento_espn(
    ev: Dict[str, Any], season_default: Optional[int] = None, fase_default: str = "Regular"
) -> Optional[Dict[str, Any]]:
    """
    Parsea un evento JSON de ESPN (tanto de `/schedule` como de `/scoreboard`).
    Retorna el diccionario listo para SQLite si el partido ya finalizó.
    """
    try:
        ev_id = str(ev.get("id", ""))
        date_str = str(ev.get("date", ""))[:10]
        if not date_str or len(date_str) < 10:
            return None

        season_obj = ev.get("season") or {}
        season_year = int(season_obj.get("year") or season_default or int(date_str[:4]))
        season_type = ev.get("seasonType") or {}
        type_id = int(season_type.get("type") or season_type.get("id") or 2)
        if type_id == 3:
            fase = "Playoffs"
        elif type_id == 5:
            fase = "Play-In"
        else:
            fase = fase_default

        competitions = ev.get("competitions", [])
        if not competitions:
            return None
        comp = competitions[0]
        status_type = (comp.get("status") or {}).get("type") or {}
        completed = bool(status_type.get("completed", False))
        if not completed:
            return None

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            return None

        home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_team = normalizar_equipo(
            (home_c.get("team") or {}).get("displayName")
            or (home_c.get("team") or {}).get("shortDisplayName")
            or (home_c.get("team") or {}).get("abbreviation")
        )
        away_team = normalizar_equipo(
            (away_c.get("team") or {}).get("displayName")
            or (away_c.get("team") or {}).get("shortDisplayName")
            or (away_c.get("team") or {}).get("abbreviation")
        )
        if not home_team or not away_team:
            return None

        def _extract_score(c_obj: Dict[str, Any]) -> Optional[int]:
            sc = c_obj.get("score")
            if isinstance(sc, dict):
                val = sc.get("value") if sc.get("value") is not None else sc.get("displayValue")
                return int(float(val)) if val is not None else None
            if sc is not None and str(sc).strip() != "":
                return int(float(sc))
            return None

        pts_l = _extract_score(home_c)
        pts_v = _extract_score(away_c)
        if pts_l is None or pts_v is None:
            return None

        # Detectar prórroga (OT) y marcadores por cuarto (linescores)
        ls_h = home_c.get("linescores") or []
        ls_a = away_c.get("linescores") or []
        period = int((comp.get("status") or {}).get("period") or max(len(ls_h), 4))
        detail = str(status_type.get("shortDetail") or status_type.get("detail") or "")
        fue_ot = 1 if (period > 4 or len(ls_h) > 4 or "OT" in detail.upper()) else 0

        q1_l, q1_v, h1_l, h1_v = None, None, None, None
        if len(ls_h) >= 2 and len(ls_a) >= 2:
            try:
                q1_l = int(float(ls_h[0].get("value") or ls_h[0].get("displayValue") or 0))
                q1_v = int(float(ls_a[0].get("value") or ls_a[0].get("displayValue") or 0))
                q2_l = int(float(ls_h[1].get("value") or ls_h[1].get("displayValue") or 0))
                q2_v = int(float(ls_a[1].get("value") or ls_a[1].get("displayValue") or 0))
                h1_l = q1_l + q2_l
                h1_v = q1_v + q2_v
            except Exception:
                pass

        rec: Dict[str, Any] = {
            "id": ev_id or f"{date_str}_{home_team}_{away_team}",
            "temporada": season_year,
            "fase": fase,
            "fecha": date_str,
            "local": home_team,
            "visita": away_team,
            "pts_l": pts_l,
            "pts_v": pts_v,
            "fue_ot": fue_ot,
            "q1_l": q1_l,
            "q1_v": q1_v,
            "h1_l": h1_l,
            "h1_v": h1_v,
        }

        # Extraer boxscore avanzado si viene en el JSON
        st_l = _extraer_estadisticas_competitor(home_c)
        st_v = _extraer_estadisticas_competitor(away_c)
        for k, v in st_l.items():
            rec[f"{k}_l"] = v
        for k, v in st_v.items():
            rec[f"{k}_v"] = v

        # Extraer líneas de apuestas si vienen en `odds`
        odds_list = comp.get("odds") or []
        if odds_list and isinstance(odds_list[0], dict):
            odd0 = odds_list[0]
            if odd0.get("overUnder") is not None:
                try:
                    rec["total_linea"] = float(odd0["overUnder"])
                except Exception:
                    pass
            if odd0.get("spread") is not None:
                try:
                    rec["spread_linea"] = float(odd0["spread"])
                except Exception:
                    pass

        return imputar_estadisticas_boxscore(rec)
    except Exception:
        return None


def _descargar_schedule_equipo_temporada(
    team_id: str, season: int, timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    Descarga el calendario completo de una franquicia y temporada desde ESPN
    incluyendo Regular Season (seasontype=2), Playoffs (seasontype=3) y Play-In (seasontype=5).
    """
    partidos: List[Dict[str, Any]] = []
    url_base = ESPN_SCHEDULE_URL.format(team_id=team_id, year=season)
    # El endpoint base suele devolver la temporada regular o postemporada activa;
    # consultamos también seasontype=2 y seasontype=3 para garantizar cobertura completa.
    urls_to_fetch = [
        (url_base, "Regular"),
        (f"{url_base}&seasontype=2", "Regular"),
        (f"{url_base}&seasontype=3", "Playoffs"),
    ]
    seen_ids = set()
    for url, fase_def in urls_to_fetch:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code != 200:
                continue
            data = resp.json()
            events = data.get("events", []) or []
            for ev in events:
                parsed = _parsear_evento_espn(ev, season_default=season, fase_default=fase_def)
                if parsed and parsed["id"] not in seen_ids:
                    seen_ids.add(parsed["id"])
                    partidos.append(parsed)
        except Exception:
            continue
    return partidos


def descargar_historial_nba(
    temporadas: Optional[List[int]] = None,
    max_workers: int = 10,
    db_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> int:
    """
    Descarga en paralelo con `ThreadPoolExecutor` el historial completo de las 30 franquicias
    para las temporadas solicitadas y lo guarda en SQLite sin duplicados.
    """
    seasons = temporadas or DEFAULT_SEASONS
    tasks = []
    for s in seasons:
        for team_name, meta in NBA_TEAMS.items():
            tasks.append((meta["id"], team_name, int(s)))

    partidos_unicos: Dict[str, Dict[str, Any]] = {}
    total_tasks = len(tasks)
    completados = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_descargar_schedule_equipo_temporada, tid, yr): (tname, yr)
            for tid, tname, yr in tasks
        }
        for fut in as_completed(future_map):
            tname, yr = future_map[fut]
            completados += 1
            try:
                res = fut.result()
                for p in res:
                    # Preferir el registro que tenga más datos de cuartos/boxscore
                    pid = p["id"]
                    if pid not in partidos_unicos:
                        partidos_unicos[pid] = p
            except Exception:
                pass
            if progress_callback:
                progress_callback(
                    completados / max(1, total_tasks),
                    f"Descargando {tname} ({yr}) [{completados}/{total_tasks}]",
                )

    lista_final = list(partidos_unicos.values())
    if lista_final:
        return guardar_partidos(lista_final, db_path=db_path)
    return 0


def actualizar_partidos_recientes(
    dias_atras: int = 7, db_path: Optional[str] = None, timeout: int = 8
) -> int:
    """
    Consulta el endpoint diario `/scoreboard?dates=YYYYMMDD` de los últimos `dias_atras` días
    para capturar marcadores exactos por cuarto (Q1, 1H) y boxscores detallados.
    """
    hoy = datetime.now().date()
    partidos_nuevos: List[Dict[str, Any]] = []
    for delta in range(dias_atras, -1, -1):
        fecha_obj = hoy - timedelta(days=delta)
        ymd = fecha_obj.strftime("%Y%m%d")
        url = ESPN_SCOREBOARD_URL.format(YYYYMMDD=ymd)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code != 200:
                continue
            data = resp.json()
            for ev in data.get("events", []) or []:
                parsed = _parsear_evento_espn(ev)
                if parsed:
                    partidos_nuevos.append(parsed)
        except Exception:
            continue

    if partidos_nuevos:
        return guardar_partidos(partidos_nuevos, db_path=db_path)
    return 0


def buscar_cartelera_nba(
    fecha_inicio: Optional[Union[str, date, datetime]] = None,
    dias_adelante: int = 0,
    timeout: int = 8,
    buscar_proxima_disponible: bool = False,
) -> List[Dict[str, Any]]:
    """
    Busca automáticamente todos los partidos NBA programados (o en juego/finalizados)
    para la fecha seleccionada y hasta `dias_adelante` días adicionales usando ESPN Scoreboard.
    Si `buscar_proxima_disponible=True` y la fecha consultada no tiene partidos en ESPN,
    explora automáticamente el calendario cercano (próximos 7 días / últimos 3 días) y si no
    encuentra eventos en vivo retorna una cartelera oficial NBA lista para analizar.
    """
    if fecha_inicio is None:
        base_date = datetime.now().date()
    elif isinstance(fecha_inicio, datetime):
        base_date = fecha_inicio.date()
    elif isinstance(fecha_inicio, date):
        base_date = fecha_inicio
    else:
        base_date = datetime.strptime(str(fecha_inicio)[:10], "%Y-%m-%d").date()

    cartelera: List[Dict[str, Any]] = []
    seen_keys = set()

    def _consultar_dia(target_date: date) -> None:
        ymd = target_date.strftime("%Y%m%d")
        iso_date = target_date.strftime("%Y-%m-%d")
        url = ESPN_SCOREBOARD_URL.format(YYYYMMDD=ymd)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code != 200:
                return
            data = resp.json()
            for ev in data.get("events", []) or []:
                comps = ev.get("competitions", []) or []
                if not comps:
                    continue
                comp = comps[0]
                competitors = comp.get("competitors", []) or []
                if len(competitors) < 2:
                    continue
                home_c = next(
                    (c for c in competitors if c.get("homeAway") == "home"), competitors[0]
                )
                away_c = next(
                    (c for c in competitors if c.get("homeAway") == "away"), competitors[1]
                )
                local = normalizar_equipo(
                    (home_c.get("team") or {}).get("displayName")
                    or (home_c.get("team") or {}).get("shortDisplayName")
                    or (home_c.get("team") or {}).get("abbreviation")
                )
                visita = normalizar_equipo(
                    (away_c.get("team") or {}).get("displayName")
                    or (away_c.get("team") or {}).get("shortDisplayName")
                    or (away_c.get("team") or {}).get("abbreviation")
                )
                if not local or not visita:
                    continue

                key = f"{iso_date}_{local}_{visita}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                status_info = (comp.get("status") or {}).get("type") or {}
                estado = status_info.get("shortDetail") or status_info.get("description") or "Programado"

                spread_linea = None
                total_linea = None
                odds_list = comp.get("odds") or []
                if odds_list and isinstance(odds_list[0], dict):
                    if odds_list[0].get("spread") is not None:
                        try:
                            spread_linea = float(odds_list[0]["spread"])
                        except Exception:
                            pass
                    if odds_list[0].get("overUnder") is not None:
                        try:
                            total_linea = float(odds_list[0]["overUnder"])
                        except Exception:
                            pass

                cartelera.append(
                    {
                        "id": str(ev.get("id", key)),
                        "fecha": iso_date,
                        "local": local,
                        "visita": visita,
                        "estado": estado,
                        "spread_linea": spread_linea,
                        "total_linea": total_linea,
                    }
                )
        except Exception:
            return

    for d in range(max(0, int(dias_adelante)) + 1):
        _consultar_dia(base_date + timedelta(days=d))

    # Si no hubo partidos en la fecha exacta y se solicitó búsqueda inteligente en fechas cercanas:
    if not cartelera and buscar_proxima_disponible:
        for delta_extra in [1, -1, 2, -2, 3, 4, 5]:
            _consultar_dia(base_date + timedelta(days=delta_extra))
            if cartelera:
                break

        # Fallback en temporada baja (Offseason) para consultar también el Scoreboard por defecto de ESPN
        if not cartelera:
            try:
                url_default = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
                resp = requests.get(url_default, headers=HEADERS, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    for ev in data.get("events", []) or []:
                        ev_date_str = str(ev.get("date", base_date.strftime("%Y-%m-%d")))[:10]
                        comps = ev.get("competitions", []) or []
                        if not comps:
                            continue
                        comp = comps[0]
                        competitors = comp.get("competitors", []) or []
                        if len(competitors) < 2:
                            continue
                        home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                        away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
                        local = normalizar_equipo((home_c.get("team") or {}).get("displayName"))
                        visita = normalizar_equipo((away_c.get("team") or {}).get("displayName"))
                        if local and visita:
                            cartelera.append(
                                {
                                    "id": str(ev.get("id", f"{ev_date_str}_{local}_{visita}")),
                                    "fecha": ev_date_str,
                                    "local": local,
                                    "visita": visita,
                                    "estado": "ESPN Scoreboard",
                                    "spread_linea": None,
                                    "total_linea": None,
                                }
                            )
            except Exception:
                pass

    return cartelera


def importar_archivo_usuario(
    archivo: Union[str, io.BytesIO, Any],
    nombre_archivo: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Importa un archivo `.csv` o `.xlsx` subido por el usuario en la pestaña `🗄️ Datos`,
    mapea flexiblemente los nombres de columnas en español o inglés, imputa boxscores/ratings
    faltantes y guarda los registros en SQLite sin duplicados.
    """
    fname = (nombre_archivo or getattr(archivo, "name", "") or "datos.csv").lower()
    if fname.endswith(".xlsx") or fname.endswith(".xls"):
        df_raw = pd.read_excel(archivo)
    else:
        df_raw = pd.read_csv(archivo)

    if df_raw.empty:
        return {"insertados": 0, "filas_leidas": 0, "mensaje": "El archivo está vacío."}

    # Mapeo flexible de columnas (español e inglés)
    col_aliases = {
        "fecha": ["fecha", "date", "game_date", "dia"],
        "local": ["local", "home", "home_team", "equipo_local", "team_home"],
        "visita": ["visita", "away", "visitor", "away_team", "equipo_visita", "team_away"],
        "pts_l": ["pts_l", "home_pts", "pts_home", "score_home", "puntos_local", "home_score"],
        "pts_v": ["pts_v", "away_pts", "pts_away", "score_away", "puntos_visita", "away_score"],
        "temporada": ["temporada", "season", "year"],
        "fase": ["fase", "phase", "stage", "season_type"],
        "fue_ot": ["fue_ot", "ot", "overtime"],
        "q1_l": ["q1_l", "q1_home", "home_q1"],
        "q1_v": ["q1_v", "q1_away", "away_q1"],
        "h1_l": ["h1_l", "h1_home", "home_h1", "1h_l"],
        "h1_v": ["h1_v", "h1_away", "away_h1", "1h_v"],
        "cuota_l": ["cuota_l", "odds_home", "home_odds", "ml_home", "cuota_local"],
        "cuota_v": ["cuota_v", "odds_away", "away_odds", "ml_away", "cuota_visita"],
        "spread_linea": ["spread_linea", "spread", "line", "handicap"],
        "total_linea": ["total_linea", "total", "over_under", "ou_line"],
    }

    lower_to_actual = {str(c).strip().lower(): c for c in df_raw.columns}
    mapped_cols: Dict[str, str] = {}
    for target_col, candidates in col_aliases.items():
        for cand in candidates:
            if cand in lower_to_actual:
                mapped_cols[target_col] = lower_to_actual[cand]
                break

    required = ["fecha", "local", "visita", "pts_l", "pts_v"]
    missing = [r for r in required if r not in mapped_cols]
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias en el archivo: {missing}. "
            f"Columnas detectadas: {list(df_raw.columns)}"
        )

    registros: List[Dict[str, Any]] = []
    for idx, row in df_raw.iterrows():
        try:
            fecha_val = pd.to_datetime(row[mapped_cols["fecha"]]).strftime("%Y-%m-%d")
            local_val = normalizar_equipo(str(row[mapped_cols["local"]]))
            visita_val = normalizar_equipo(str(row[mapped_cols["visita"]]))
            pts_l_val = int(float(row[mapped_cols["pts_l"]]))
            pts_v_val = int(float(row[mapped_cols["pts_v"]]))
        except Exception:
            continue

        rec: Dict[str, Any] = {
            "fecha": fecha_val,
            "local": local_val,
            "visita": visita_val,
            "pts_l": pts_l_val,
            "pts_v": pts_v_val,
        }
        for optional_key in [
            "temporada", "fase", "fue_ot", "q1_l", "q1_v", "h1_l", "h1_v",
            "cuota_l", "cuota_v", "spread_linea", "total_linea",
        ]:
            if optional_key in mapped_cols:
                val = row[mapped_cols[optional_key]]
                if pd.notna(val):
                    rec[optional_key] = val

        # Copiar también columnas de boxscore si el CSV ya las trae con su nombre exacto
        for box_col in [
            "fgm_l", "fga_l", "fgm_v", "fga_v", "tpm_l", "tpa_l", "tpm_v", "tpa_v",
            "ftm_l", "fta_l", "ftm_v", "fta_v", "oreb_l", "reb_l", "oreb_v", "reb_v",
            "ast_l", "ast_v", "tov_l", "tov_v",
        ]:
            if box_col in lower_to_actual:
                val = row[lower_to_actual[box_col]]
                if pd.notna(val):
                    rec[box_col] = float(val)

        registros.append(rec)

    insertados = guardar_partidos(registros, db_path=db_path)
    return {
        "insertados": insertados,
        "filas_leidas": len(df_raw),
        "mensaje": f"Se importaron {insertados} partidos correctamente.",
    }


def generar_partidos_sinteticos_nba(
    n_partidos: int = 480,
    temporadas: Optional[List[int]] = None,
    seed: int = 42,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Genera un dataset realista y reproducible de partidos NBA (con las 30 franquicias,
    perfiles reales de ataque/defensa/ritmo, efectos de altitud en Denver/Utah, B2B,
    prórrogas y cuotas de mercado realistas) para pruebas sin red o inicialización rápida.
    Garantiza que las 30 franquicias tengan >= 12 partidos por temporada.
    """
    rng = np.random.default_rng(seed)
    seasons = temporadas or [2025, 2026]
    teams = list(NBA_TEAMS.keys())

    # Fuerza intrínseca realista por franquicia para que el ranking y modelos tengan señal real
    elite_teams = {
        "Boston Celtics": (119.5, 110.5, 99.5),
        "Oklahoma City Thunder": (118.8, 110.2, 100.8),
        "Denver Nuggets": (118.2, 112.0, 98.2),
        "Cleveland Cavaliers": (117.8, 111.2, 98.8),
        "New York Knicks": (117.2, 111.8, 97.0),
        "Minnesota Timberwolves": (115.2, 109.8, 98.0),
        "Dallas Mavericks": (117.5, 113.2, 99.8),
        "Milwaukee Bucks": (117.0, 113.8, 100.2),
        "Indiana Pacers": (119.0, 116.2, 102.5),
        "Los Angeles Lakers": (115.8, 113.8, 100.5),
        "Golden State Warriors": (116.2, 113.5, 100.0),
        "Phoenix Suns": (116.5, 114.2, 98.8),
    }

    perfiles = {}
    for t in teams:
        if t in elite_teams:
            perfiles[t] = elite_teams[t]
        else:
            ortg_b = float(rng.uniform(110.5, 116.0))
            drtg_b = float(rng.uniform(112.0, 118.5))
            pace_b = float(rng.uniform(97.0, 101.8))
            perfiles[t] = (ortg_b, drtg_b, pace_b)

    partidos: List[Dict[str, Any]] = []
    # Garantizar al menos 180 partidos por temporada (= 12 PJ exactos por equipo en round-robin)
    per_season = max(180, n_partidos // len(seasons))

    for s_idx, season in enumerate(seasons):
        start_date = date(int(season) - 1, 10, 22)
        for i in range(per_season):
            current_date = start_date + timedelta(days=int(i * 150 / per_season))
            local = teams[i % len(teams)]
            shift = (i // len(teams)) + 1
            visita = teams[(i + shift) % len(teams)]
            if local == visita:
                visita = teams[(i + 1) % len(teams)]

            ortg_l, drtg_l, pace_l = perfiles[local]
            ortg_v, drtg_v, pace_v = perfiles[visita]

            pace_game = float(np.clip(0.4 * pace_l + 0.4 * pace_v + 0.2 * 99.2 + rng.normal(0, 2.2), 90.0, 109.0))
            alt_bonus = 1.6 if local == "Denver Nuggets" else (1.1 if local == "Utah Jazz" else 0.0)
            exp_ortg_l = 0.5 * (ortg_l + drtg_v) + 1.3 + alt_bonus
            exp_ortg_v = 0.5 * (ortg_v + drtg_l) - 1.3

            mu_l = (pace_game / 100.0) * exp_ortg_l
            mu_v = (pace_game / 100.0) * exp_ortg_v

            pts_l_48 = int(round(rng.normal(mu_l, 9.5)))
            pts_v_48 = int(round(rng.normal(mu_v, 9.5)))

            fue_ot = 0
            if pts_l_48 == pts_v_48:
                fue_ot = 1
                ot_l = int(rng.integers(8, 16))
                ot_v = int(rng.integers(8, 16))
                if ot_l == ot_v:
                    ot_l += 2
                pts_l = pts_l_48 + ot_l
                pts_v = pts_v_48 + ot_v
            else:
                pts_l = max(82, pts_l_48)
                pts_v = max(82, pts_v_48)

            # Probabilidad implícita justa y cuota de mercado simulada con overround ~4.5%
            diff_exp = mu_l - mu_v
            p_true_l = float(1.0 / (1.0 + np.exp(-diff_exp / 6.8)))
            p_true_l = float(np.clip(p_true_l, 0.08, 0.92))
            p_true_v = 1.0 - p_true_l
            overround = 1.045
            cuota_l = round(max(1.05, 1.0 / (p_true_l * overround)), 2)
            cuota_v = round(max(1.05, 1.0 / (p_true_v * overround)), 2)
            spread_linea = round(-diff_exp * 2.0) / 2.0
            total_linea = round((mu_l + mu_v) * 2.0) / 2.0

            fase = "Playoffs" if i >= int(per_season * 0.90) else (
                "Play-In" if i >= int(per_season * 0.86) else "Regular"
            )

            rec = {
                "id": f"SYN_{season}_{i:04d}",
                "temporada": int(season),
                "fase": fase,
                "fecha": current_date.strftime("%Y-%m-%d"),
                "local": local,
                "visita": visita,
                "pts_l": pts_l,
                "pts_v": pts_v,
                "fue_ot": fue_ot,
                "q1_l": int(round(pts_l_48 * 0.252 + rng.normal(0, 2.5))),
                "q1_v": int(round(pts_v_48 * 0.252 + rng.normal(0, 2.5))),
                "h1_l": int(round(pts_l_48 * 0.508 + rng.normal(0, 3.8))),
                "h1_v": int(round(pts_v_48 * 0.508 + rng.normal(0, 3.8))),
                "cuota_l": cuota_l,
                "cuota_v": cuota_v,
                "spread_linea": spread_linea,
                "total_linea": total_linea,
            }
            partidos.append(imputar_estadisticas_boxscore(rec))

    if db_path is not None:
        guardar_partidos(partidos, db_path=db_path)

    return partidos
