"""
Configuración central de NBA ULTIMATE AI:
- Rutas de base de datos y directorios
- Constantes matemáticas por defecto (Elo, Pace, Ratings, Half-Life, Monte Carlo)
- Catálogo completo de las 30 franquicias NBA con IDs de ESPN, abreviaturas,
  conferencias, divisiones, coordenadas/husos horarios, altitud de pabellones (msnm)
  y perfil de arranque (Q1/1H).
- Endpoints JSON públicos de ESPN NBA (100% libres de API keys).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(DATA_DIR / "nba_ultimate.db")

# Endpoints públicos de ESPN NBA (Sin API Key)
ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ESPN_SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule?season={year}"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={YYYYMMDD}"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"

# Temporadas soportadas por defecto para descarga histórica (últimas 4 temporadas)
DEFAULT_SEASONS = [2023, 2024, 2025, 2026]

# Constantes del Sistema Elo NBA (Estilo FiveThirtyEight)
ELO_BASE = 1500.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 75.0  # ~2.68 puntos de ventaja local base en Elo (75 / 28)
ELO_SPREAD_DIVISOR = 28.0  # diff_elo / 28.0 = spread_elo
ELO_SEASON_REGRESSION = 0.25  # 25% de regresión hacia 1500 al cambiar de temporada

# Constantes del Modelo Posesional y Eficiencia NBA
NBA_AVG_PACE = 99.2        # Posesiones promedio por equipo en 48 min
NBA_AVG_ORTG = 114.8       # Puntos por 100 posesiones promedio NBA
NBA_HOME_COURT_PTS = 2.5   # Ventaja de localía base en puntos (48 min)
DEFAULT_HALF_LIFE = 120.0  # Decaimiento exponencial en días
DEFAULT_MC_SIMS = 7500     # Simulaciones Monte Carlo por defecto (entre 5,000 y 10,000)
BASE_STD_DEV_DIFF = 11.8   # Desviación estándar base del diferencial de puntos NBA
BASE_STD_DEV_TOTAL = 16.5  # Desviación estándar base del total de puntos NBA

# Proporciones estándar por periodo
PROP_Q1_BASE = 0.252       # 1er Cuarto ~ 25.2% del puntaje total
PROP_1H_BASE = 0.508       # 1ra Mitad (Descanso) ~ 50.8% del puntaje total

# Factores tácticos de Fatiga y Calendario NBA
FATIGA_B2B_VISITA_PTS = -2.6   # Back-to-Back visitante
FATIGA_B2B_LOCAL_PTS = -1.5    # Back-to-Back local
FATIGA_DENSIDAD_PTS = -1.2     # 3-in-4 o 4-in-6 noches
BONO_DESCANSO_PTS = 1.0        # Rest Advantage (>=3 días vs 1 día)

# Altitud de Pabellones NBA (msnm) y bonos fisiológicos locales
ALTITUD_NBA: Dict[str, Dict[str, Any]] = {
    "Denver Nuggets": {
        "arena": "Ball Arena",
        "altitud_msnm": 1609,
        "bono_pts": 1.6,
        "intensificador_fatiga": 1.40,  # +40% adicional si visitante viene en B2B o 3-in-4
    },
    "Utah Jazz": {
        "arena": "Delta Center",
        "altitud_msnm": 1288,
        "bono_pts": 1.1,
        "intensificador_fatiga": 1.40,  # +40% adicional si visitante viene en B2B o 3-in-4
    },
}

# Catálogo completo de las 30 Franquicias NBA
# Incluye ID oficial de ESPN, abreviatura, conferencia, división, coordenadas, huso horario,
# altitud del pabellón (msnm) y factores de perfil de arranque Q1/1H.
NBA_TEAMS: Dict[str, Dict[str, Any]] = {
    "Atlanta Hawks": {
        "id": "1", "abbr": "ATL", "conf": "East", "div": "Southeast",
        "lat": 33.7573, "lon": -84.3963, "tz": -5, "altitud": 320,
        "arena": "State Farm Arena", "q1_mult": 1.01, "h1_mult": 1.01
    },
    "Boston Celtics": {
        "id": "2", "abbr": "BOS", "conf": "East", "div": "Atlantic",
        "lat": 42.3662, "lon": -71.0621, "tz": -5, "altitud": 14,
        "arena": "TD Garden", "q1_mult": 1.03, "h1_mult": 1.02
    },
    "New Orleans Pelicans": {
        "id": "3", "abbr": "NOP", "conf": "West", "div": "Southwest",
        "lat": 29.9490, "lon": -90.0821, "tz": -6, "altitud": 2,
        "arena": "Smoothie King Center", "q1_mult": 1.00, "h1_mult": 1.00
    },
    "Chicago Bulls": {
        "id": "4", "abbr": "CHI", "conf": "East", "div": "Central",
        "lat": 41.8807, "lon": -87.6742, "tz": -6, "altitud": 181,
        "arena": "United Center", "q1_mult": 0.98, "h1_mult": 0.99
    },
    "Cleveland Cavaliers": {
        "id": "5", "abbr": "CLE", "conf": "East", "div": "Central",
        "lat": 41.4965, "lon": -81.6882, "tz": -5, "altitud": 199,
        "arena": "Rocket Mortgage FieldHouse", "q1_mult": 1.02, "h1_mult": 1.01
    },
    "Dallas Mavericks": {
        "id": "6", "abbr": "DAL", "conf": "West", "div": "Southwest",
        "lat": 32.7905, "lon": -96.8103, "tz": -6, "altitud": 131,
        "arena": "American Airlines Center", "q1_mult": 1.01, "h1_mult": 1.01
    },
    "Denver Nuggets": {
        "id": "7", "abbr": "DEN", "conf": "West", "div": "Northwest",
        "lat": 39.7487, "lon": -105.0077, "tz": -7, "altitud": 1609,
        "arena": "Ball Arena", "q1_mult": 1.02, "h1_mult": 1.01
    },
    "Detroit Pistons": {
        "id": "8", "abbr": "DET", "conf": "East", "div": "Central",
        "lat": 42.3410, "lon": -83.0551, "tz": -5, "altitud": 183,
        "arena": "Little Caesars Arena", "q1_mult": 0.99, "h1_mult": 0.99
    },
    "Golden State Warriors": {
        "id": "9", "abbr": "GSW", "conf": "West", "div": "Pacific",
        "lat": 37.7680, "lon": -122.3877, "tz": -8, "altitud": 5,
        "arena": "Chase Center", "q1_mult": 1.01, "h1_mult": 1.01
    },
    "Houston Rockets": {
        "id": "10", "abbr": "HOU", "conf": "West", "div": "Southwest",
        "lat": 29.7508, "lon": -95.3621, "tz": -6, "altitud": 15,
        "arena": "Toyota Center", "q1_mult": 1.00, "h1_mult": 1.00
    },
    "Indiana Pacers": {
        "id": "11", "abbr": "IND", "conf": "East", "div": "Central",
        "lat": 39.7639, "lon": -86.1555, "tz": -5, "altitud": 218,
        "arena": "Gainbridge Fieldhouse", "q1_mult": 1.03, "h1_mult": 1.02
    },
    "LA Clippers": {
        "id": "12", "abbr": "LAC", "conf": "West", "div": "Pacific",
        "lat": 33.9441, "lon": -118.3422, "tz": -8, "altitud": 40,
        "arena": "Intuit Dome", "q1_mult": 1.00, "h1_mult": 1.00
    },
    "Los Angeles Lakers": {
        "id": "13", "abbr": "LAL", "conf": "West", "div": "Pacific",
        "lat": 34.0430, "lon": -118.2673, "tz": -8, "altitud": 71,
        "arena": "Crypto.com Arena", "q1_mult": 1.00, "h1_mult": 1.00
    },
    "Miami Heat": {
        "id": "14", "abbr": "MIA", "conf": "East", "div": "Southeast",
        "lat": 25.7814, "lon": -80.1870, "tz": -5, "altitud": 2,
        "arena": "Kaseya Center", "q1_mult": 0.98, "h1_mult": 0.99
    },
    "Milwaukee Bucks": {
        "id": "15", "abbr": "MIL", "conf": "East", "div": "Central",
        "lat": 43.0451, "lon": -87.9172, "tz": -6, "altitud": 188,
        "arena": "Fiserv Forum", "q1_mult": 1.01, "h1_mult": 1.01
    },
    "Minnesota Timberwolves": {
        "id": "16", "abbr": "MIN", "conf": "West", "div": "Northwest",
        "lat": 44.9795, "lon": -93.2760, "tz": -6, "altitud": 253,
        "arena": "Target Center", "q1_mult": 1.01, "h1_mult": 1.00
    },
    "Brooklyn Nets": {
        "id": "17", "abbr": "BKN", "conf": "East", "div": "Atlantic",
        "lat": 40.6826, "lon": -73.9754, "tz": -5, "altitud": 18,
        "arena": "Barclays Center", "q1_mult": 0.99, "h1_mult": 0.99
    },
    "New York Knicks": {
        "id": "18", "abbr": "NYK", "conf": "East", "div": "Atlantic",
        "lat": 40.7505, "lon": -73.9934, "tz": -5, "altitud": 11,
        "arena": "Madison Square Garden", "q1_mult": 1.01, "h1_mult": 1.01
    },
    "Orlando Magic": {
        "id": "19", "abbr": "ORL", "conf": "East", "div": "Southeast",
        "lat": 28.5392, "lon": -81.3839, "tz": -5, "altitud": 25,
        "arena": "Kia Center", "q1_mult": 0.99, "h1_mult": 0.99
    },
    "Philadelphia 76ers": {
        "id": "20", "abbr": "PHI", "conf": "East", "div": "Atlantic",
        "lat": 39.9012, "lon": -75.1720, "tz": -5, "altitud": 12,
        "arena": "Wells Fargo Center", "q1_mult": 1.01, "h1_mult": 1.01
    },
    "Phoenix Suns": {
        "id": "21", "abbr": "PHX", "conf": "West", "div": "Pacific",
        "lat": 33.4457, "lon": -112.0712, "tz": -7, "altitud": 331,
        "arena": "Footprint Center", "q1_mult": 1.01, "h1_mult": 1.01
    },
    "Portland Trail Blazers": {
        "id": "22", "abbr": "POR", "conf": "West", "div": "Northwest",
        "lat": 45.5316, "lon": -122.6668, "tz": -8, "altitud": 15,
        "arena": "Moda Center", "q1_mult": 0.98, "h1_mult": 0.99
    },
    "Sacramento Kings": {
        "id": "23", "abbr": "SAC", "conf": "West", "div": "Pacific",
        "lat": 38.5802, "lon": -121.4997, "tz": -8, "altitud": 9,
        "arena": "Golden 1 Center", "q1_mult": 1.02, "h1_mult": 1.01
    },
    "San Antonio Spurs": {
        "id": "24", "abbr": "SAS", "conf": "West", "div": "Southwest",
        "lat": 29.4270, "lon": -98.4375, "tz": -6, "altitud": 198,
        "arena": "Frost Bank Center", "q1_mult": 0.99, "h1_mult": 1.00
    },
    "Oklahoma City Thunder": {
        "id": "25", "abbr": "OKC", "conf": "West", "div": "Northwest",
        "lat": 35.4634, "lon": -97.5151, "tz": -6, "altitud": 366,
        "arena": "Paycom Center", "q1_mult": 1.02, "h1_mult": 1.02
    },
    "Utah Jazz": {
        "id": "26", "abbr": "UTA", "conf": "West", "div": "Northwest",
        "lat": 40.7683, "lon": -111.9011, "tz": -7, "altitud": 1288,
        "arena": "Delta Center", "q1_mult": 1.00, "h1_mult": 1.00
    },
    "Washington Wizards": {
        "id": "27", "abbr": "WAS", "conf": "East", "div": "Southeast",
        "lat": 38.8981, "lon": -77.0209, "tz": -5, "altitud": 7,
        "arena": "Capital One Arena", "q1_mult": 0.98, "h1_mult": 0.99
    },
    "Toronto Raptors": {
        "id": "28", "abbr": "TOR", "conf": "East", "div": "Atlantic",
        "lat": 43.6435, "lon": -79.3791, "tz": -5, "altitud": 76,
        "arena": "Scotiabank Arena", "q1_mult": 0.99, "h1_mult": 1.00
    },
    "Memphis Grizzlies": {
        "id": "29", "abbr": "MEM", "conf": "West", "div": "Southwest",
        "lat": 35.1382, "lon": -90.0505, "tz": -6, "altitud": 103,
        "arena": "FedExForum", "q1_mult": 1.00, "h1_mult": 1.00
    },
    "Charlotte Hornets": {
        "id": "30", "abbr": "CHA", "conf": "East", "div": "Southeast",
        "lat": 35.2251, "lon": -80.8392, "tz": -5, "altitud": 229,
        "arena": "Spectrum Center", "q1_mult": 0.98, "h1_mult": 0.99
    },
}

# Índices de búsqueda rápida y normalización de nombres de equipos
TEAM_ABBR_TO_NAME: Dict[str, str] = {}
TEAM_ID_TO_NAME: Dict[str, str] = {}
TEAM_ALIASES: Dict[str, str] = {}

for _name, _meta in NBA_TEAMS.items():
    TEAM_ABBR_TO_NAME[_meta["abbr"].upper()] = _name
    TEAM_ID_TO_NAME[str(_meta["id"])] = _name
    TEAM_ALIASES[_name.lower()] = _name
    TEAM_ALIASES[_meta["abbr"].lower()] = _name
    # Apodo corto (ej. "Lakers", "Celtics", "76ers", "Trail Blazers")
    short_nickname = _name.split()[-1].lower()
    TEAM_ALIASES[short_nickname] = _name

# Alias adicionales comunes en ESPN y casas de apuestas (Betano / Betsafe)
_EXTRA_ALIASES = {
    "los angeles clippers": "LA Clippers",
    "l.a. clippers": "LA Clippers",
    "la clippers": "LA Clippers",
    "clippers": "LA Clippers",
    "l.a. lakers": "Los Angeles Lakers",
    "la lakers": "Los Angeles Lakers",
    "lakers": "Los Angeles Lakers",
    "gs warriors": "Golden State Warriors",
    "golden state": "Golden State Warriors",
    "gsw": "Golden State Warriors",
    "ny knicks": "New York Knicks",
    "knicks": "New York Knicks",
    "nyk": "New York Knicks",
    "no pelicans": "New Orleans Pelicans",
    "pelicans": "New Orleans Pelicans",
    "nop": "New Orleans Pelicans",
    "no": "New Orleans Pelicans",
    "sa spurs": "San Antonio Spurs",
    "spurs": "San Antonio Spurs",
    "sas": "San Antonio Spurs",
    "sa": "San Antonio Spurs",
    "okc thunder": "Oklahoma City Thunder",
    "oklahoma city": "Oklahoma City Thunder",
    "thunder": "Oklahoma City Thunder",
    "okc": "Oklahoma City Thunder",
    "phila 76ers": "Philadelphia 76ers",
    "sixers": "Philadelphia 76ers",
    "76ers": "Philadelphia 76ers",
    "blazers": "Portland Trail Blazers",
    "trail blazers": "Portland Trail Blazers",
    "wolves": "Minnesota Timberwolves",
    "timberwolves": "Minnesota Timberwolves",
    "mavs": "Dallas Mavericks",
    "cavs": "Cleveland Cavaliers",
    "utah": "Utah Jazz",
    "uta": "Utah Jazz",
    "utah jazz": "Utah Jazz",
    "washington": "Washington Wizards",
    "wsh": "Washington Wizards",
    "was": "Washington Wizards",
    "phoenix": "Phoenix Suns",
    "pho": "Phoenix Suns",
    "phx": "Phoenix Suns",
    "brooklyn": "Brooklyn Nets",
    "bkn": "Brooklyn Nets",
    "brk": "Brooklyn Nets",
    "charlotte": "Charlotte Hornets",
    "cho": "Charlotte Hornets",
    "cha": "Charlotte Hornets",
}
TEAM_ALIASES.update(_EXTRA_ALIASES)


def normalizar_equipo(nombre: Optional[str]) -> str:
    """
    Normaliza cualquier variante de nombre o abreviatura de franquicia NBA
    al nombre canónico oficial de las 30 franquicias.
    """
    if not nombre:
        return ""
    limpio = str(nombre).strip()
    if limpio in NBA_TEAMS:
        return limpio
    key = limpio.lower()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    for canon_name in NBA_TEAMS:
        if key in canon_name.lower() or canon_name.lower() in key:
            return canon_name
    return limpio


def obtener_altitud_equipo(equipo_local: str) -> Dict[str, Any]:
    """
    Retorna la información de altitud de un equipo local.
    """
    canon = normalizar_equipo(equipo_local)
    if canon in ALTITUD_NBA:
        return ALTITUD_NBA[canon]
    meta = NBA_TEAMS.get(canon, {})
    return {
        "arena": meta.get("arena", "NBA Arena"),
        "altitud_msnm": meta.get("altitud", 100),
        "bono_pts": 0.0,
        "intensificador_fatiga": 1.0,
    }
