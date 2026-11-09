"""
Módulo Cuantitativo de Estadísticas de Jugadores en Tiempo Real (ESPN NBA),
Player Props (`PTS`, `REB`, `AST`, `3PM`, `PRA`) y Simulador Automático de Bajas / Lesiones (`engine/players.py`).

Arquitectura de Datos Híbrida de Alta Fiabilidad (100% libre de API Keys):
1. **Sincronización en Vivo desde ESPN NBA (`sincronizar_jugadores_espn`)**:
   - Consulta en paralelo (`ThreadPoolExecutor`) las plantillas oficiales y el estado médico
     real (`Active`, `Day-To-Day`, `Out`) de las 30 franquicias en:
     `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster`
   - Consulta los boxscores individuales de los últimos partidos jugados desde:
     `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}`
   - Calcula para cada jugador:
     * Promedios de temporada (`MIN`, `PTS`, `REB`, `AST`, `3PM`, `PRA`)
     * Forma reciente en los **últimos 5 partidos (`L5`)**: `pts_l5`, `reb_l5`, `ast_l5`, `tpm_l5`, `pra_l5`
     * Desviación estándar empírica real de cada jugador (`sd_pts`, `sd_reb`, `sd_ast`)
     * Estado oficial de lesión (`Activa`, `Duda (GTD)`, `Baja (Out)`) y detalle médico.
   - Guarda todo en la tabla SQLite `jugadores_stats`.
2. **Ponderación Cuantitativa (65% Temporada + 35% Racha Últimos 5 Partidos `L5`)**:
   - Captura cambios recientes de rotación, rachas calientes de tiro y aumentos de minutos.
3. **Respaldo Calibrado Offline (`BASE_ROSTERS_NBA`)**:
   - Si no hay red o aún no se ha ejecutado la sincronización en vivo, el motor opera sin
     interrupciones con el catálogo pre-calibrado de las 30 franquicias.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import pandas as pd
import requests

from engine.config import (
    NBA_TEAMS,
    NBA_AVG_PACE,
    NBA_AVG_ORTG,
    ESPN_SCOREBOARD_URL,
    ESPN_SUMMARY_URL,
    normalizar_equipo,
)
from engine.db import get_connection, init_db

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

ESPN_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"

# Catálogo base pre-calibrado de estrellas y titulares de las 30 franquicias NBA (Fallback Offline)
BASE_ROSTERS_NBA: Dict[str, List[Dict[str, Any]]] = {
    "Atlanta Hawks": [
        {"jugador": "Trae Young", "pos": "PG", "min": 35.8, "pts": 26.2, "reb": 3.1, "ast": 10.8, "tpm": 3.2, "usg": 30.5, "impacto_pts": 4.2},
        {"jugador": "Jalen Johnson", "pos": "PF", "min": 34.5, "pts": 19.5, "reb": 9.8, "ast": 5.1, "tpm": 1.6, "usg": 22.4, "impacto_pts": 2.8},
        {"jugador": "Dyson Daniels", "pos": "SG", "min": 32.0, "pts": 13.8, "reb": 5.2, "ast": 4.1, "tpm": 1.4, "usg": 17.5, "impacto_pts": 1.9},
        {"jugador": "Onyeka Okongwu", "pos": "C", "min": 28.5, "pts": 13.4, "reb": 8.6, "ast": 2.0, "tpm": 0.7, "usg": 18.0, "impacto_pts": 1.6},
    ],
    "Boston Celtics": [
        {"jugador": "Jayson Tatum", "pos": "PF", "min": 36.2, "pts": 28.4, "reb": 8.6, "ast": 5.6, "tpm": 3.6, "usg": 31.2, "impacto_pts": 5.2},
        {"jugador": "Jaylen Brown", "pos": "SG", "min": 34.8, "pts": 24.5, "reb": 6.1, "ast": 4.4, "tpm": 2.3, "usg": 28.8, "impacto_pts": 3.6},
        {"jugador": "Derrick White", "pos": "PG", "min": 33.2, "pts": 16.8, "reb": 4.4, "ast": 4.8, "tpm": 3.3, "usg": 19.5, "impacto_pts": 2.9},
        {"jugador": "Kristaps Porzingis", "pos": "C", "min": 29.5, "pts": 19.8, "reb": 7.1, "ast": 2.0, "tpm": 2.2, "usg": 23.5, "impacto_pts": 2.7},
        {"jugador": "Jrue Holiday", "pos": "PG", "min": 31.0, "pts": 12.4, "reb": 4.6, "ast": 4.4, "tpm": 1.9, "usg": 16.2, "impacto_pts": 2.1},
    ],
    "Brooklyn Nets": [
        {"jugador": "Cam Thomas", "pos": "SG", "min": 33.4, "pts": 24.2, "reb": 3.4, "ast": 3.6, "tpm": 2.7, "usg": 30.8, "impacto_pts": 2.9},
        {"jugador": "Cameron Johnson", "pos": "SF", "min": 32.2, "pts": 18.8, "reb": 4.4, "ast": 3.0, "tpm": 3.1, "usg": 21.5, "impacto_pts": 2.4},
        {"jugador": "Nic Claxton", "pos": "C", "min": 29.5, "pts": 11.8, "reb": 9.2, "ast": 2.2, "tpm": 0.0, "usg": 16.0, "impacto_pts": 2.0},
    ],
    "Charlotte Hornets": [
        {"jugador": "LaMelo Ball", "pos": "PG", "min": 34.0, "pts": 27.5, "reb": 5.4, "ast": 7.8, "tpm": 4.1, "usg": 33.5, "impacto_pts": 4.1},
        {"jugador": "Brandon Miller", "pos": "SF", "min": 33.8, "pts": 21.2, "reb": 5.0, "ast": 3.6, "tpm": 3.4, "usg": 26.2, "impacto_pts": 2.6},
        {"jugador": "Miles Bridges", "pos": "PF", "min": 32.5, "pts": 19.6, "reb": 7.4, "ast": 3.4, "tpm": 2.1, "usg": 24.0, "impacto_pts": 2.1},
    ],
    "Chicago Bulls": [
        {"jugador": "Coby White", "pos": "PG", "min": 34.5, "pts": 19.8, "reb": 4.2, "ast": 5.1, "tpm": 3.0, "usg": 23.8, "impacto_pts": 2.5},
        {"jugador": "Nikola Vucevic", "pos": "C", "min": 32.4, "pts": 19.2, "reb": 10.2, "ast": 3.4, "tpm": 1.8, "usg": 22.6, "impacto_pts": 2.4},
        {"jugador": "Josh Giddey", "pos": "SG", "min": 30.2, "pts": 13.8, "reb": 7.6, "ast": 6.8, "tpm": 1.3, "usg": 20.4, "impacto_pts": 1.9},
    ],
    "Cleveland Cavaliers": [
        {"jugador": "Donovan Mitchell", "pos": "SG", "min": 34.2, "pts": 25.4, "reb": 4.8, "ast": 5.2, "tpm": 3.5, "usg": 30.4, "impacto_pts": 4.6},
        {"jugador": "Darius Garland", "pos": "PG", "min": 31.8, "pts": 20.6, "reb": 2.8, "ast": 6.8, "tpm": 2.9, "usg": 25.2, "impacto_pts": 3.1},
        {"jugador": "Evan Mobley", "pos": "PF", "min": 31.5, "pts": 18.4, "reb": 9.4, "ast": 3.0, "tpm": 1.0, "usg": 21.8, "impacto_pts": 3.4},
        {"jugador": "Jarrett Allen", "pos": "C", "min": 30.2, "pts": 14.5, "reb": 10.4, "ast": 2.1, "tpm": 0.0, "usg": 17.2, "impacto_pts": 2.5},
    ],
    "Dallas Mavericks": [
        {"jugador": "Kyrie Irving", "pos": "PG", "min": 35.4, "pts": 25.2, "reb": 4.9, "ast": 5.4, "tpm": 3.1, "usg": 28.5, "impacto_pts": 4.2},
        {"jugador": "Anthony Davis", "pos": "C", "min": 35.0, "pts": 25.8, "reb": 11.8, "ast": 3.6, "tpm": 0.8, "usg": 28.2, "impacto_pts": 4.7},
        {"jugador": "Klay Thompson", "pos": "SF", "min": 28.8, "pts": 14.2, "reb": 3.5, "ast": 2.1, "tpm": 3.1, "usg": 20.2, "impacto_pts": 1.7},
        {"jugador": "PJ Washington", "pos": "PF", "min": 32.0, "pts": 13.6, "reb": 7.8, "ast": 2.2, "tpm": 1.7, "usg": 17.8, "impacto_pts": 2.0},
    ],
    "Denver Nuggets": [
        {"jugador": "Nikola Jokic", "pos": "C", "min": 36.8, "pts": 29.6, "reb": 12.8, "ast": 10.2, "tpm": 1.9, "usg": 31.5, "impacto_pts": 6.8},
        {"jugador": "Jamal Murray", "pos": "PG", "min": 34.5, "pts": 21.2, "reb": 4.1, "ast": 6.4, "tpm": 2.4, "usg": 26.4, "impacto_pts": 3.2},
        {"jugador": "Michael Porter Jr.", "pos": "SF", "min": 33.4, "pts": 18.4, "reb": 7.0, "ast": 2.2, "tpm": 2.8, "usg": 21.0, "impacto_pts": 2.2},
        {"jugador": "Aaron Gordon", "pos": "PF", "min": 31.5, "pts": 14.2, "reb": 6.2, "ast": 3.2, "tpm": 1.1, "usg": 17.8, "impacto_pts": 2.3},
    ],
    "Detroit Pistons": [
        {"jugador": "Cade Cunningham", "pos": "PG", "min": 35.6, "pts": 25.2, "reb": 6.4, "ast": 9.2, "tpm": 2.3, "usg": 31.0, "impacto_pts": 4.4},
        {"jugador": "Jaden Ivey", "pos": "SG", "min": 30.5, "pts": 17.6, "reb": 4.1, "ast": 4.0, "tpm": 2.0, "usg": 24.2, "impacto_pts": 2.0},
        {"jugador": "Jalen Duren", "pos": "C", "min": 28.8, "pts": 12.4, "reb": 10.6, "ast": 2.5, "tpm": 0.0, "usg": 16.8, "impacto_pts": 2.1},
    ],
    "Golden State Warriors": [
        {"jugador": "Stephen Curry", "pos": "PG", "min": 33.5, "pts": 26.4, "reb": 4.6, "ast": 6.2, "tpm": 4.6, "usg": 30.8, "impacto_pts": 5.5},
        {"jugador": "Jimmy Butler", "pos": "SF", "min": 33.2, "pts": 20.4, "reb": 5.6, "ast": 5.4, "tpm": 0.9, "usg": 23.5, "impacto_pts": 3.5},
        {"jugador": "Draymond Green", "pos": "PF", "min": 29.4, "pts": 9.0, "reb": 6.4, "ast": 6.0, "tpm": 1.2, "usg": 14.5, "impacto_pts": 2.8},
        {"jugador": "Buddy Hield", "pos": "SG", "min": 25.5, "pts": 13.2, "reb": 3.4, "ast": 1.8, "tpm": 2.9, "usg": 19.8, "impacto_pts": 1.4},
    ],
    "Houston Rockets": [
        {"jugador": "Alperen Sengun", "pos": "C", "min": 32.8, "pts": 19.6, "reb": 10.4, "ast": 5.0, "tpm": 0.4, "usg": 26.2, "impacto_pts": 3.5},
        {"jugador": "Jalen Green", "pos": "SG", "min": 33.4, "pts": 21.4, "reb": 4.5, "ast": 3.2, "tpm": 2.8, "usg": 27.6, "impacto_pts": 2.6},
        {"jugador": "Amen Thompson", "pos": "SF", "min": 31.2, "pts": 14.2, "reb": 8.4, "ast": 3.8, "tpm": 0.4, "usg": 18.4, "impacto_pts": 2.7},
        {"jugador": "Fred VanVleet", "pos": "PG", "min": 35.0, "pts": 15.2, "reb": 3.9, "ast": 6.2, "tpm": 2.7, "usg": 19.6, "impacto_pts": 2.5},
    ],
    "Indiana Pacers": [
        {"jugador": "Tyrese Haliburton", "pos": "PG", "min": 34.4, "pts": 19.2, "reb": 3.8, "ast": 9.6, "tpm": 3.1, "usg": 23.6, "impacto_pts": 4.2},
        {"jugador": "Pascal Siakam", "pos": "PF", "min": 33.6, "pts": 20.8, "reb": 7.4, "ast": 3.4, "tpm": 1.6, "usg": 24.4, "impacto_pts": 3.3},
        {"jugador": "Myles Turner", "pos": "C", "min": 30.5, "pts": 15.8, "reb": 6.8, "ast": 1.6, "tpm": 2.1, "usg": 20.2, "impacto_pts": 2.4},
        {"jugador": "Bennedict Mathurin", "pos": "SG", "min": 29.8, "pts": 16.4, "reb": 5.4, "ast": 2.0, "tpm": 1.6, "usg": 22.0, "impacto_pts": 1.7},
    ],
    "LA Clippers": [
        {"jugador": "James Harden", "pos": "PG", "min": 34.8, "pts": 21.6, "reb": 5.8, "ast": 8.6, "tpm": 2.9, "usg": 27.8, "impacto_pts": 4.0},
        {"jugador": "Kawhi Leonard", "pos": "SF", "min": 32.5, "pts": 21.8, "reb": 5.9, "ast": 3.5, "tpm": 2.0, "usg": 26.8, "impacto_pts": 4.1},
        {"jugador": "Norman Powell", "pos": "SG", "min": 32.8, "pts": 22.8, "reb": 3.4, "ast": 2.2, "tpm": 3.4, "usg": 25.4, "impacto_pts": 2.8},
        {"jugador": "Ivica Zubac", "pos": "C", "min": 32.2, "pts": 15.6, "reb": 12.2, "ast": 2.5, "tpm": 0.0, "usg": 19.5, "impacto_pts": 2.7},
    ],
    "Los Angeles Lakers": [
        {"jugador": "Luka Doncic", "pos": "PG", "min": 36.4, "pts": 28.8, "reb": 8.4, "ast": 8.2, "tpm": 3.6, "usg": 33.8, "impacto_pts": 5.6},
        {"jugador": "LeBron James", "pos": "PF", "min": 35.0, "pts": 24.2, "reb": 7.6, "ast": 8.6, "tpm": 2.2, "usg": 28.6, "impacto_pts": 4.5},
        {"jugador": "Austin Reaves", "pos": "SG", "min": 34.2, "pts": 18.8, "reb": 4.4, "ast": 5.8, "tpm": 2.5, "usg": 22.4, "impacto_pts": 2.6},
        {"jugador": "Rui Hachimura", "pos": "SF", "min": 31.0, "pts": 13.4, "reb": 5.2, "ast": 1.5, "tpm": 1.6, "usg": 17.5, "impacto_pts": 1.6},
    ],
    "Memphis Grizzlies": [
        {"jugador": "Ja Morant", "pos": "PG", "min": 32.5, "pts": 23.2, "reb": 4.6, "ast": 7.8, "tpm": 1.6, "usg": 30.4, "impacto_pts": 4.0},
        {"jugador": "Jaren Jackson Jr.", "pos": "PF", "min": 31.2, "pts": 22.6, "reb": 6.2, "ast": 2.2, "tpm": 2.0, "usg": 27.2, "impacto_pts": 3.6},
        {"jugador": "Desmond Bane", "pos": "SG", "min": 32.4, "pts": 19.4, "reb": 5.2, "ast": 5.0, "tpm": 2.6, "usg": 24.2, "impacto_pts": 2.8},
    ],
    "Miami Heat": [
        {"jugador": "Tyler Herro", "pos": "SG", "min": 35.2, "pts": 23.8, "reb": 5.4, "ast": 5.4, "tpm": 3.6, "usg": 28.4, "impacto_pts": 3.6},
        {"jugador": "Bam Adebayo", "pos": "C", "min": 34.4, "pts": 17.8, "reb": 10.0, "ast": 4.2, "tpm": 0.8, "usg": 22.6, "impacto_pts": 3.5},
        {"jugador": "Andrew Wiggins", "pos": "SF", "min": 31.5, "pts": 17.2, "reb": 4.6, "ast": 2.4, "tpm": 2.1, "usg": 21.4, "impacto_pts": 2.1},
    ],
    "Milwaukee Bucks": [
        {"jugador": "Giannis Antetokounmpo", "pos": "PF", "min": 35.4, "pts": 31.2, "reb": 11.8, "ast": 6.2, "tpm": 0.4, "usg": 35.2, "impacto_pts": 6.2},
        {"jugador": "Damian Lillard", "pos": "PG", "min": 35.8, "pts": 25.4, "reb": 4.6, "ast": 7.2, "tpm": 3.4, "usg": 28.8, "impacto_pts": 3.9},
        {"jugador": "Brook Lopez", "pos": "C", "min": 31.2, "pts": 12.6, "reb": 5.1, "ast": 1.7, "tpm": 1.9, "usg": 16.4, "impacto_pts": 2.0},
        {"jugador": "Kyle Kuzma", "pos": "SF", "min": 31.0, "pts": 15.8, "reb": 6.0, "ast": 2.6, "tpm": 1.5, "usg": 22.0, "impacto_pts": 1.6},
    ],
    "Minnesota Timberwolves": [
        {"jugador": "Anthony Edwards", "pos": "SG", "min": 36.2, "pts": 27.6, "reb": 5.8, "ast": 4.6, "tpm": 3.9, "usg": 31.6, "impacto_pts": 4.9},
        {"jugador": "Julius Randle", "pos": "PF", "min": 33.2, "pts": 19.6, "reb": 7.2, "ast": 4.6, "tpm": 1.6, "usg": 24.8, "impacto_pts": 2.6},
        {"jugador": "Rudy Gobert", "pos": "C", "min": 33.0, "pts": 11.4, "reb": 10.8, "ast": 1.8, "tpm": 0.0, "usg": 14.2, "impacto_pts": 3.1},
        {"jugador": "Naz Reid", "pos": "PF", "min": 27.5, "pts": 14.6, "reb": 5.8, "ast": 2.0, "tpm": 2.2, "usg": 21.0, "impacto_pts": 2.0},
    ],
    "New Orleans Pelicans": [
        {"jugador": "Zion Williamson", "pos": "PF", "min": 31.5, "pts": 24.2, "reb": 7.2, "ast": 5.1, "tpm": 0.2, "usg": 31.2, "impacto_pts": 3.8},
        {"jugador": "Trey Murphy III", "pos": "SF", "min": 34.8, "pts": 21.6, "reb": 5.2, "ast": 3.4, "tpm": 3.3, "usg": 24.6, "impacto_pts": 2.8},
        {"jugador": "CJ McCollum", "pos": "SG", "min": 33.2, "pts": 20.8, "reb": 3.8, "ast": 4.1, "tpm": 3.2, "usg": 25.8, "impacto_pts": 2.3},
    ],
    "New York Knicks": [
        {"jugador": "Jalen Brunson", "pos": "PG", "min": 35.6, "pts": 26.4, "reb": 3.2, "ast": 7.4, "tpm": 2.5, "usg": 30.6, "impacto_pts": 4.8},
        {"jugador": "Karl-Anthony Towns", "pos": "C", "min": 35.0, "pts": 24.8, "reb": 13.0, "ast": 3.2, "tpm": 2.1, "usg": 27.4, "impacto_pts": 4.2},
        {"jugador": "Mikal Bridges", "pos": "SF", "min": 37.2, "pts": 17.8, "reb": 3.4, "ast": 3.6, "tpm": 2.3, "usg": 19.8, "impacto_pts": 2.4},
        {"jugador": "OG Anunoby", "pos": "PF", "min": 36.2, "pts": 16.8, "reb": 4.8, "ast": 2.2, "tpm": 2.2, "usg": 19.2, "impacto_pts": 2.8},
        {"jugador": "Josh Hart", "pos": "SG", "min": 37.5, "pts": 14.2, "reb": 9.6, "ast": 5.8, "tpm": 1.1, "usg": 15.8, "impacto_pts": 2.6},
    ],
    "Oklahoma City Thunder": [
        {"jugador": "Shai Gilgeous-Alexander", "pos": "PG", "min": 34.6, "pts": 32.2, "reb": 5.2, "ast": 6.2, "tpm": 2.1, "usg": 34.2, "impacto_pts": 6.4},
        {"jugador": "Jalen Williams", "pos": "SF", "min": 33.2, "pts": 21.4, "reb": 5.4, "ast": 5.2, "tpm": 1.8, "usg": 26.2, "impacto_pts": 3.5},
        {"jugador": "Chet Holmgren", "pos": "C", "min": 30.8, "pts": 17.2, "reb": 8.6, "ast": 2.4, "tpm": 1.6, "usg": 21.6, "impacto_pts": 3.6},
        {"jugador": "Isaiah Hartenstein", "pos": "C", "min": 28.8, "pts": 11.4, "reb": 11.2, "ast": 4.0, "tpm": 0.0, "usg": 15.2, "impacto_pts": 2.3},
    ],
    "Orlando Magic": [
        {"jugador": "Paolo Banchero", "pos": "PF", "min": 34.8, "pts": 24.4, "reb": 7.2, "ast": 5.0, "tpm": 1.8, "usg": 31.2, "impacto_pts": 3.8},
        {"jugador": "Franz Wagner", "pos": "SF", "min": 34.2, "pts": 24.2, "reb": 5.6, "ast": 4.8, "tpm": 1.9, "usg": 29.8, "impacto_pts": 3.9},
        {"jugador": "Jalen Suggs", "pos": "PG", "min": 29.5, "pts": 16.2, "reb": 4.1, "ast": 3.8, "tpm": 2.2, "usg": 23.0, "impacto_pts": 2.6},
    ],
    "Philadelphia 76ers": [
        {"jugador": "Tyrese Maxey", "pos": "PG", "min": 37.4, "pts": 27.2, "reb": 3.5, "ast": 6.1, "tpm": 3.3, "usg": 30.8, "impacto_pts": 4.4},
        {"jugador": "Joel Embiid", "pos": "C", "min": 32.0, "pts": 26.8, "reb": 9.2, "ast": 4.2, "tpm": 1.3, "usg": 34.5, "impacto_pts": 5.2},
        {"jugador": "Paul George", "pos": "SF", "min": 32.8, "pts": 17.4, "reb": 5.4, "ast": 4.4, "tpm": 2.4, "usg": 23.6, "impacto_pts": 2.7},
    ],
    "Phoenix Suns": [
        {"jugador": "Kevin Durant", "pos": "PF", "min": 36.2, "pts": 27.1, "reb": 6.2, "ast": 4.3, "tpm": 2.5, "usg": 29.4, "impacto_pts": 4.6},
        {"jugador": "Devin Booker", "pos": "SG", "min": 36.8, "pts": 26.4, "reb": 4.1, "ast": 6.8, "tpm": 2.6, "usg": 29.8, "impacto_pts": 4.2},
        {"jugador": "Bradley Beal", "pos": "SG", "min": 32.2, "pts": 17.4, "reb": 3.5, "ast": 3.6, "tpm": 2.0, "usg": 22.0, "impacto_pts": 1.9},
    ],
    "Portland Trail Blazers": [
        {"jugador": "Anfernee Simons", "pos": "PG", "min": 33.2, "pts": 19.4, "reb": 2.8, "ast": 4.9, "tpm": 3.1, "usg": 25.6, "impacto_pts": 2.3},
        {"jugador": "Shaedon Sharpe", "pos": "SG", "min": 31.8, "pts": 18.2, "reb": 4.4, "ast": 2.8, "tpm": 2.1, "usg": 24.4, "impacto_pts": 1.9},
        {"jugador": "Deandre Ayton", "pos": "C", "min": 30.5, "pts": 14.6, "reb": 10.2, "ast": 1.6, "tpm": 0.2, "usg": 19.6, "impacto_pts": 2.0},
    ],
    "Sacramento Kings": [
        {"jugador": "Domantas Sabonis", "pos": "C", "min": 35.8, "pts": 19.8, "reb": 14.1, "ast": 6.4, "tpm": 0.9, "usg": 22.4, "impacto_pts": 4.3},
        {"jugador": "Zach LaVine", "pos": "SG", "min": 34.4, "pts": 23.2, "reb": 4.6, "ast": 4.2, "tpm": 3.1, "usg": 26.8, "impacto_pts": 2.8},
        {"jugador": "DeMar DeRozan", "pos": "SF", "min": 35.2, "pts": 22.1, "reb": 4.0, "ast": 4.0, "tpm": 1.0, "usg": 25.4, "impacto_pts": 2.6},
        {"jugador": "Malik Monk", "pos": "PG", "min": 31.6, "pts": 17.6, "reb": 3.8, "ast": 5.8, "tpm": 2.3, "usg": 24.2, "impacto_pts": 2.1},
    ],
    "San Antonio Spurs": [
        {"jugador": "Victor Wembanyama", "pos": "C", "min": 33.4, "pts": 24.6, "reb": 11.0, "ast": 3.8, "tpm": 3.1, "usg": 31.4, "impacto_pts": 5.5},
        {"jugador": "De'Aaron Fox", "pos": "PG", "min": 35.6, "pts": 24.8, "reb": 4.8, "ast": 6.4, "tpm": 2.1, "usg": 29.2, "impacto_pts": 3.9},
        {"jugador": "Devin Vassell", "pos": "SG", "min": 31.2, "pts": 16.4, "reb": 3.8, "ast": 3.0, "tpm": 2.4, "usg": 22.4, "impacto_pts": 2.0},
    ],
    "Toronto Raptors": [
        {"jugador": "Scottie Barnes", "pos": "PF", "min": 34.6, "pts": 20.2, "reb": 8.0, "ast": 6.4, "tpm": 1.4, "usg": 26.4, "impacto_pts": 3.6},
        {"jugador": "RJ Barrett", "pos": "SF", "min": 33.4, "pts": 21.6, "reb": 6.4, "ast": 5.6, "tpm": 1.8, "usg": 27.8, "impacto_pts": 2.5},
        {"jugador": "Immanuel Quickley", "pos": "PG", "min": 31.2, "pts": 17.2, "reb": 3.8, "ast": 6.2, "tpm": 2.5, "usg": 23.6, "impacto_pts": 2.3},
    ],
    "Utah Jazz": [
        {"jugador": "Lauri Markkanen", "pos": "PF", "min": 32.4, "pts": 19.8, "reb": 6.4, "ast": 2.0, "tpm": 2.9, "usg": 23.8, "impacto_pts": 3.2},
        {"jugador": "Collin Sexton", "pos": "SG", "min": 29.5, "pts": 18.6, "reb": 2.8, "ast": 4.2, "tpm": 1.8, "usg": 25.4, "impacto_pts": 2.1},
        {"jugador": "John Collins", "pos": "C", "min": 29.8, "pts": 18.2, "reb": 8.4, "ast": 2.2, "tpm": 1.4, "usg": 22.8, "impacto_pts": 2.0},
        {"jugador": "Keyonte George", "pos": "PG", "min": 31.2, "pts": 16.4, "reb": 3.6, "ast": 5.8, "tpm": 2.5, "usg": 24.6, "impacto_pts": 1.6},
    ],
    "Washington Wizards": [
        {"jugador": "Jordan Poole", "pos": "PG", "min": 31.4, "pts": 21.0, "reb": 3.1, "ast": 4.8, "tpm": 3.4, "usg": 28.4, "impacto_pts": 2.2},
        {"jugador": "Bilal Coulibaly", "pos": "SF", "min": 33.2, "pts": 12.6, "reb": 5.2, "ast": 3.4, "tpm": 1.1, "usg": 17.8, "impacto_pts": 1.7},
        {"jugador": "Alex Sarr", "pos": "C", "min": 27.6, "pts": 11.8, "reb": 6.8, "ast": 2.2, "tpm": 1.4, "usg": 20.4, "impacto_pts": 1.5},
    ],
}


def _normalizar_jugador_base(p: Dict[str, Any], equipo: str) -> Dict[str, Any]:
    pts = float(p.get("pts", 15.0))
    reb = float(p.get("reb", 5.0))
    ast = float(p.get("ast", 3.5))
    tpm = float(p.get("tpm", 1.8))
    pra = round(pts + reb + ast, 1)
    pts_l5 = float(p.get("pts_l5") if p.get("pts_l5") is not None else pts)
    reb_l5 = float(p.get("reb_l5") if p.get("reb_l5") is not None else reb)
    ast_l5 = float(p.get("ast_l5") if p.get("ast_l5") is not None else ast)
    tpm_l5 = float(p.get("tpm_l5") if p.get("tpm_l5") is not None else tpm)
    pra_l5 = round(pts_l5 + reb_l5 + ast_l5, 1)
    usg = float(p.get("usg") or min(35.0, max(14.0, pts * 0.95 + ast * 0.6)))
    impacto = float(p.get("impacto_pts") or round(max(1.2, (pts * 0.12 + reb * 0.08 + ast * 0.14) - 0.5), 1))

    return {
        "id": str(p.get("id") or f"{equipo}_{p['jugador']}").replace(" ", "_"),
        "jugador": str(p["jugador"]),
        "equipo": normalizar_equipo(equipo),
        "pos": str(p.get("pos") or "G/F"),
        "pj": int(p.get("pj") or 20),
        "min": round(float(p.get("min", 31.0)), 1),
        "pts": round(pts, 1),
        "reb": round(reb, 1),
        "ast": round(ast, 1),
        "tpm": round(tpm, 1),
        "pra": pra,
        "pts_l5": round(pts_l5, 1),
        "reb_l5": round(reb_l5, 1),
        "ast_l5": round(ast_l5, 1),
        "tpm_l5": round(tpm_l5, 1),
        "pra_l5": pra_l5,
        "sd_pts": round(float(p.get("sd_pts") or max(3.2, 0.26 * pts)), 2),
        "sd_reb": round(float(p.get("sd_reb") or max(1.6, 0.30 * reb)), 2),
        "sd_ast": round(float(p.get("sd_ast") or max(1.4, 0.32 * ast)), 2),
        "usg": round(usg, 1),
        "impacto_pts": round(impacto, 1),
        "estado_lesion": str(p.get("estado_lesion") or "Activa"),
        "detalle_lesion": str(p.get("detalle_lesion") or ""),
        "fuente": str(p.get("fuente") or "Catálogo Base Calibrado"),
        "actualizado_en": str(p.get("actualizado_en") or datetime.now().strftime("%Y-%m-%d")),
    }


def guardar_jugadores_db(
    jugadores: List[Dict[str, Any]], db_path: Optional[str] = None
) -> int:
    """Guarda o actualiza estadísticas y estados de lesión de jugadores en SQLite (`jugadores_stats`)."""
    if not jugadores:
        return 0
    init_db(db_path)
    conn = get_connection(db_path)
    cols = [
        "id", "jugador", "equipo", "pos", "pj", "min", "pts", "reb", "ast", "tpm", "pra",
        "pts_l5", "reb_l5", "ast_l5", "tpm_l5", "pra_l5", "sd_pts", "sd_reb", "sd_ast",
        "usg", "impacto_pts", "estado_lesion", "detalle_lesion", "fuente", "actualizado_en",
    ]
    placeholders = ", ".join(["?"] * len(cols))
    update_clause = ", ".join([f"{c}=excluded.{c}" for c in cols if c != "id"])
    guardados = 0
    try:
        cur = conn.cursor()
        for raw in jugadores:
            eq = normalizar_equipo(raw.get("equipo"))
            if not eq or not raw.get("jugador"):
                continue
            item = _normalizar_jugador_base(raw, eq)
            cur.execute(
                "SELECT id FROM jugadores_stats WHERE jugador = ? AND equipo = ?",
                (item["jugador"], item["equipo"]),
            )
            ex = cur.fetchone()
            if ex:
                item["id"] = ex["id"]
            cur.execute(
                f"""
                INSERT INTO jugadores_stats ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT(id) DO UPDATE SET {update_clause}
                """,
                [item[c] for c in cols],
            )
            guardados += 1
        conn.commit()
    finally:
        conn.close()
    return guardados


def _descargar_roster_y_lesiones_equipo(
    team_id: str, team_name: str, timeout: int = 8
) -> Dict[str, Dict[str, Any]]:
    """
    Descarga la plantilla oficial y el estado de lesión en vivo (`Active`, `Day-To-Day`, `Out`)
    desde el endpoint público de ESPN:
    `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster`
    """
    url = ESPN_ROSTER_URL.format(team_id=team_id)
    out: Dict[str, Dict[str, Any]] = {}
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return out
        data = resp.json()
        athletes = data.get("athletes", []) or []
        for ath in athletes:
            # En NBA ESPN, `athletes` puede ser una lista directa de objetos jugador o grupos por posición
            items = ath.get("items", [ath]) if isinstance(ath, dict) and "items" in ath else [ath]
            for p in items:
                if not isinstance(p, dict):
                    continue
                nombre = str(p.get("displayName") or p.get("fullName") or "").strip()
                if not nombre:
                    continue
                pos = str((p.get("position") or {}).get("abbreviation") or "G/F")
                injuries = p.get("injuries") or []
                estado_lesion = "Activa"
                detalle_lesion = ""
                if injuries and isinstance(injuries[0], dict):
                    st_raw = str(injuries[0].get("status") or "").lower()
                    det_raw = str(injuries[0].get("shortComment") or injuries[0].get("longComment") or st_raw)
                    if "out" in st_raw or "injured" in st_raw or "suspension" in st_raw:
                        estado_lesion = "Baja (Out)"
                    elif "day" in st_raw or "questionable" in st_raw or "doubtful" in st_raw:
                        estado_lesion = "Duda (GTD)"
                    detalle_lesion = det_raw[:120]

                out[nombre] = {
                    "id": str(p.get("id") or f"{team_name}_{nombre}"),
                    "jugador": nombre,
                    "equipo": team_name,
                    "pos": pos,
                    "estado_lesion": estado_lesion,
                    "detalle_lesion": detalle_lesion,
                }
    except Exception:
        pass
    return out


def _extraer_boxscore_jugadores_summary(event_id: str, timeout: int = 8) -> List[Dict[str, Any]]:
    """
    Consulta el endpoint público `/summary?event={event_id}` de ESPN NBA y extrae las
    actuaciones individuales reales (`MIN`, `PTS`, `REB`, `AST`, `3PM`) de todos los jugadores
    que participaron en ese partido.
    """
    url = ESPN_SUMMARY_URL.format(event_id=event_id)
    actuaciones: List[Dict[str, Any]] = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return actuaciones
        data = resp.json()
        box = data.get("boxscore") or {}
        players_blocks = box.get("players") or []
        for team_block in players_blocks:
            t_info = team_block.get("team") or {}
            equipo = normalizar_equipo(
                t_info.get("displayName") or t_info.get("shortDisplayName") or t_info.get("abbreviation")
            )
            if not equipo:
                continue
            stats_groups = team_block.get("statistics") or []
            if not stats_groups:
                continue
            grp = stats_groups[0]
            labels = [str(k).upper() for k in (grp.get("labels") or grp.get("names") or [])]
            athletes = grp.get("athletes") or []
            for ath_entry in athletes:
                if ath_entry.get("didNotPlay"):
                    continue
                ath_obj = ath_entry.get("athlete") or {}
                nombre = str(ath_obj.get("displayName") or ath_obj.get("shortName") or "").strip()
                pos = str((ath_obj.get("position") or {}).get("abbreviation") or "G/F")
                raw_stats = ath_entry.get("stats") or []
                if not nombre or not raw_stats or len(raw_stats) < len(labels):
                    continue
                s_map = dict(zip(labels, raw_stats))
                try:
                    min_str = str(s_map.get("MIN", "0")).split(":")[0]
                    minutos = float(min_str) if min_str.replace(".", "", 1).isdigit() else 0.0
                    if minutos < 5.0:
                        continue
                    pts = float(s_map.get("PTS", 0))
                    reb = float(s_map.get("REB", 0))
                    ast = float(s_map.get("AST", 0))
                    threept_str = str(s_map.get("3PT", "0-0"))
                    tpm = float(threept_str.split("-")[0]) if "-" in threept_str else float(threept_str or 0)
                    actuaciones.append(
                        {
                            "jugador": nombre,
                            "equipo": equipo,
                            "pos": pos,
                            "min": minutos,
                            "pts": pts,
                            "reb": reb,
                            "ast": ast,
                            "tpm": tpm,
                        }
                    )
                except Exception:
                    continue
    except Exception:
        pass
    return actuaciones


def sincronizar_jugadores_espn(
    dias_boxscores: int = 14,
    max_workers: int = 10,
    db_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Sincroniza en vivo desde ESPN NBA (100% sin API Keys):
      1. Plantillas actuales y reporte oficial de lesionados (`Active`, `GTD`, `Out`) de las 30 franquicias.
      2. Boxscores individuales reales de los últimos `dias_boxscores` días para calcular la forma
         reciente (`L5`: últimos 5 partidos) y la desviación estándar empírica de cada jugador.
      3. Combina las estadísticas reales de ESPN con los priors de temporada y persiste todo en
         la tabla SQLite `jugadores_stats`.
    """
    init_db(db_path)
    rosters_espn: Dict[str, Dict[str, Dict[str, Any]]] = {}
    team_items = list(NBA_TEAMS.items())

    # Paso 1: Descargar plantillas y estado de lesiones en paralelo para las 30 franquicias
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_roster = {
            ex.submit(_descargar_roster_y_lesiones_equipo, meta["id"], tname): tname
            for tname, meta in team_items
        }
        done_r = 0
        for fut in as_completed(fut_roster):
            tname = fut_roster[fut]
            done_r += 1
            try:
                rosters_espn[tname] = fut.result()
            except Exception:
                rosters_espn[tname] = {}
            if progress_callback:
                progress_callback(
                    0.45 * (done_r / len(team_items)),
                    f"Sincronizando plantilla y parte médico de {tname} ({done_r}/30)...",
                )

    # Paso 2: Recolectar IDs de partidos recientes finalizados desde ESPN Scoreboard o SQLite
    event_ids: List[str] = []
    try:
        conn = get_connection(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM partidos WHERE id NOT LIKE 'SYN_%' ORDER BY fecha DESC LIMIT 90"
        )
        event_ids = [str(r["id"]) for r in cur.fetchall() if str(r["id"]).isdigit()]
        conn.close()
    except Exception:
        pass

    if len(event_ids) < 25:
        hoy = datetime.now().date()
        for d in range(max(1, int(dias_boxscores))):
            ymd = (hoy - timedelta(days=d)).strftime("%Y%m%d")
            try:
                resp = requests.get(ESPN_SCOREBOARD_URL.format(YYYYMMDD=ymd), headers=HEADERS, timeout=6)
                if resp.status_code == 200:
                    for ev in resp.json().get("events", []) or []:
                        eid = str(ev.get("id", ""))
                        if eid.isdigit() and eid not in event_ids:
                            event_ids.append(eid)
            except Exception:
                continue

    # Paso 3: Descargar boxscores individuales reales en paralelo
    logs_por_jugador: Dict[str, List[Dict[str, Any]]] = {}
    if event_ids:
        target_events = event_ids[:75]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_box = {ex.submit(_extraer_boxscore_jugadores_summary, eid): eid for eid in target_events}
            done_b = 0
            for fut in as_completed(fut_box):
                done_b += 1
                try:
                    acts = fut.result()
                    for a in acts:
                        key_j = f"{a['equipo']}::{a['jugador']}"
                        logs_por_jugador.setdefault(key_j, []).append(a)
                except Exception:
                    pass
                if progress_callback:
                    progress_callback(
                        0.45 + 0.50 * (done_b / len(target_events)),
                        f"Procesando boxscores reales de jugadores ESPN ({done_b}/{len(target_events)})...",
                    )

    # Paso 4: Construir registros consolidados (Temporada + Racha L5 + Estado Médico en vivo)
    ahora_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    registros_finales: List[Dict[str, Any]] = []
    lesionados_detectados = 0

    for equipo in NBA_TEAMS:
        base_map = {p["jugador"]: dict(p) for p in BASE_ROSTERS_NBA.get(equipo, [])}
        espn_map = rosters_espn.get(equipo, {})

        # Unir jugadores del catálogo base con los jugadores activos/lesionados reportados por ESPN
        nombres_unidos = list(base_map.keys())
        for n_espn in espn_map:
            if n_espn not in base_map:
                # Si tiene logs recientes en boxscore con >= 18 min, lo agregamos automáticamente
                logs_n = logs_por_jugador.get(f"{equipo}::{n_espn}", [])
                if logs_n and np.mean([x["min"] for x in logs_n]) >= 18.0:
                    nombres_unidos.append(n_espn)

        for nombre in nombres_unidos:
            p_base = base_map.get(nombre, {"jugador": nombre, "pos": "G/F", "min": 24.0, "pts": 12.0, "reb": 4.2, "ast": 2.6, "tpm": 1.3})
            info_med = espn_map.get(nombre, {})
            logs = logs_por_jugador.get(f"{equipo}::{nombre}", [])

            estado_les = info_med.get("estado_lesion", "Activa")
            detalle_les = info_med.get("detalle_lesion", "")
            if estado_les != "Activa":
                lesionados_detectados += 1

            if logs:
                ult5 = logs[:5]
                pts_l5 = float(np.mean([x["pts"] for x in ult5]))
                reb_l5 = float(np.mean([x["reb"] for x in ult5]))
                ast_l5 = float(np.mean([x["ast"] for x in ult5]))
                tpm_l5 = float(np.mean([x["tpm"] for x in ult5]))
                min_esp = 0.60 * float(p_base.get("min", 30.0)) + 0.40 * float(np.mean([x["min"] for x in logs]))
                pts_temp = 0.55 * float(p_base.get("pts", pts_l5)) + 0.45 * float(np.mean([x["pts"] for x in logs]))
                reb_temp = 0.55 * float(p_base.get("reb", reb_l5)) + 0.45 * float(np.mean([x["reb"] for x in logs]))
                ast_temp = 0.55 * float(p_base.get("ast", ast_l5)) + 0.45 * float(np.mean([x["ast"] for x in logs]))
                tpm_temp = 0.55 * float(p_base.get("tpm", tpm_l5)) + 0.45 * float(np.mean([x["tpm"] for x in logs]))
                sd_pts = float(np.std([x["pts"] for x in logs])) if len(logs) >= 4 else max(3.2, 0.26 * pts_temp)
                sd_reb = float(np.std([x["reb"] for x in logs])) if len(logs) >= 4 else max(1.6, 0.30 * reb_temp)
                sd_ast = float(np.std([x["ast"] for x in logs])) if len(logs) >= 4 else max(1.4, 0.32 * ast_temp)
                fuente_str = f"ESPN Live ({len(logs)} juegos)"
                pj_val = max(len(logs), 15)
            else:
                pts_temp = float(p_base["pts"])
                reb_temp = float(p_base["reb"])
                ast_temp = float(p_base["ast"])
                tpm_temp = float(p_base["tpm"])
                min_esp = float(p_base["min"])
                pts_l5, reb_l5, ast_l5, tpm_l5 = pts_temp, reb_temp, ast_temp, tpm_temp
                sd_pts = max(3.2, 0.26 * pts_temp)
                sd_reb = max(1.6, 0.30 * reb_temp)
                sd_ast = max(1.4, 0.32 * ast_temp)
                fuente_str = "ESPN Roster Live" if info_med else "Catálogo Base Calibrado"
                pj_val = 25

            registros_finales.append(
                {
                    "id": info_med.get("id") or f"{equipo}_{nombre}".replace(" ", "_"),
                    "jugador": nombre,
                    "equipo": equipo,
                    "pos": info_med.get("pos") or p_base.get("pos", "G/F"),
                    "pj": pj_val,
                    "min": min_esp,
                    "pts": pts_temp,
                    "reb": reb_temp,
                    "ast": ast_temp,
                    "tpm": tpm_temp,
                    "pts_l5": pts_l5,
                    "reb_l5": reb_l5,
                    "ast_l5": ast_l5,
                    "tpm_l5": tpm_l5,
                    "sd_pts": max(2.5, sd_pts),
                    "sd_reb": max(1.2, sd_reb),
                    "sd_ast": max(1.1, sd_ast),
                    "usg": p_base.get("usg"),
                    "impacto_pts": p_base.get("impacto_pts"),
                    "estado_lesion": estado_les,
                    "detalle_lesion": detalle_les,
                    "fuente": fuente_str,
                    "actualizado_en": ahora_str,
                }
            )

    total_guardados = guardar_jugadores_db(registros_finales, db_path=db_path)
    if progress_callback:
        progress_callback(1.0, f"¡Sincronización de {total_guardados} jugadores completada!")

    return {
        "jugadores_actualizados": total_guardados,
        "lesionados_detectados": lesionados_detectados,
        "partidos_boxscore_analizados": len(event_ids[:75]),
        "actualizado_en": ahora_str,
    }


def obtener_jugadores_equipo(
    equipo: str, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Obtiene los jugadores del equipo desde SQLite (`jugadores_stats`).
    Si la tabla aún está vacía para ese equipo, la inicializa automáticamente con el
    catálogo calibrado y retorna la lista completa.
    """
    canon = normalizar_equipo(equipo)
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM jugadores_stats WHERE equipo = ? ORDER BY pts DESC",
            (canon,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        if rows:
            return rows
    except Exception:
        pass
    finally:
        conn.close()

    base_list = [_normalizar_jugador_base(p, canon) for p in BASE_ROSTERS_NBA.get(canon, [])]
    if base_list:
        guardar_jugadores_db(base_list, db_path=db_path)
    return base_list


def obtener_bajas_detectadas_equipo(
    equipo: str, db_path: Optional[str] = None
) -> List[str]:
    """
    Retorna la lista de jugadores cuyo `estado_lesion` en vivo es `Baja (Out)`
    para pre-cargarlos automáticamente en el Simulador de Bajas.
    """
    jugs = obtener_jugadores_equipo(equipo, db_path=db_path)
    return [j["jugador"] for j in jugs if "Baja" in str(j.get("estado_lesion", ""))]


def obtener_catalogo_completo_jugadores(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Retorna un DataFrame con todos los jugadores principales de las 30 franquicias NBA,
    mostrando sus medias de temporada, su racha reciente `L5 (últimos 5 juegos)`,
    su estado médico oficial (`Activa / Duda GTD / Baja Out`) y la fuente de datos.
    """
    filas: List[Dict[str, Any]] = []
    for eq in NBA_TEAMS:
        for p in obtener_jugadores_equipo(eq, db_path=db_path):
            filas.append(
                {
                    "Jugador": p["jugador"],
                    "Equipo": p["equipo"],
                    "Pos": p["pos"],
                    "Estado": p.get("estado_lesion", "Activa"),
                    "MIN": round(float(p["min"]), 1),
                    "PTS": round(float(p["pts"]), 1),
                    "PTS (L5)": round(float(p.get("pts_l5", p["pts"])), 1),
                    "REB": round(float(p["reb"]), 1),
                    "AST": round(float(p["ast"]), 1),
                    "3PM": round(float(p["tpm"]), 1),
                    "PRA (PTS+REB+AST)": round(float(p["pra"]), 1),
                    "PRA (L5)": round(float(p.get("pra_l5", p["pra"])), 1),
                    "USG%": round(float(p["usg"]), 1),
                    "Impacto Spread (pts)": round(float(p["impacto_pts"]), 1),
                    "Fuente": p.get("fuente", "Catálogo Base"),
                }
            )
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.sort_values(by=["PRA (PTS+REB+AST)", "PTS"], ascending=[False, False]).reset_index(drop=True)
    return df


def calcular_impacto_bajas(
    equipo_local: str,
    equipo_visita: str,
    bajas_local: Optional[List[str]] = None,
    bajas_visita: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calcula el impacto en puntos sobre el Spread y el Total del partido cuando uno o más
    jugadores clave de `equipo_local` o `equipo_visita` son marcados como Baja / Ausentes.
    """
    canon_l = normalizar_equipo(equipo_local)
    canon_v = normalizar_equipo(equipo_visita)
    bajas_l_set = set(bajas_local or [])
    bajas_v_set = set(bajas_visita or [])

    roster_l = {p["jugador"]: p for p in obtener_jugadores_equipo(canon_l, db_path=db_path)}
    roster_v = {p["jugador"]: p for p in obtener_jugadores_equipo(canon_v, db_path=db_path)}

    perdida_l = sum(float(roster_l[j]["impacto_pts"]) for j in bajas_l_set if j in roster_l)
    perdida_v = sum(float(roster_v[j]["impacto_pts"]) for j in bajas_v_set if j in roster_v)

    ajuste_spread_local = round(perdida_v - perdida_l, 2)
    ajuste_total_pts = round(-0.55 * (perdida_l + perdida_v), 2)

    return {
        "bajas_local": list(bajas_l_set),
        "bajas_visita": list(bajas_v_set),
        "penalizacion_local_pts": round(-perdida_l, 2),
        "penalizacion_visita_pts": round(-perdida_v, 2),
        "ajuste_spread_local_pts": ajuste_spread_local,
        "ajuste_total_pts": ajuste_total_pts,
    }


def proyectar_y_simular_jugador(
    jugador_info: Dict[str, Any],
    es_local: bool = True,
    pace_esperado: float = NBA_AVG_PACE,
    drtg_rival: float = NBA_AVG_ORTG,
    en_b2b: bool = False,
    bajas_companeros_usg: float = 0.0,
    lineas_custom: Optional[Dict[str, float]] = None,
    n_sims: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Proyecta y simula por Monte Carlo correlacionado los 5 mercados principales de Player Props:
      1. `PTS` (Puntos)
      2. `REB` (Rebotes)
      3. `AST` (Asistencias)
      4. `3PM` (Triples Anotados)
      5. `PRA` (`PTS + REB + AST`)

    Fórmula de Proyección Híbrida Cuantitativa:
      - Base Ponderada = **65% Promedio Temporada + 35% Forma Reciente `L5` (Últimos 5 Juegos)**
      - Multiplicadores contextuales: Ritmo (`Pace`), Defensa Rival (`DRtg`), Localía (`+1.8%`),
        Fatiga `B2B` (`-4.5%`) y Redistribución de `Usage%` por bajas de compañeros.
    """
    rng = np.random.default_rng(seed)
    n_sims = int(np.clip(n_sims, 1000, 15000))

    f_pace = float(np.clip(pace_esperado / NBA_AVG_PACE, 0.90, 1.12))
    f_def = float(np.clip(drtg_rival / NBA_AVG_ORTG, 0.88, 1.14))
    f_home = 1.018 if es_local else 0.988
    f_b2b = 0.955 if en_b2b else 1.00
    f_usg_boost = float(np.clip(1.0 + 0.0065 * float(bajas_companeros_usg), 1.0, 1.22))

    mult_ofensivo = f_pace * f_def * f_home * f_b2b * f_usg_boost
    mult_rebotes = f_pace * (1.01 if es_local else 0.995) * (0.97 if en_b2b else 1.0) * (1.0 + 0.003 * float(bajas_companeros_usg))

    # Ponderación 65% Temporada + 35% Racha reciente L5
    pts_base = 0.65 * float(jugador_info["pts"]) + 0.35 * float(jugador_info.get("pts_l5", jugador_info["pts"]))
    reb_base = 0.65 * float(jugador_info["reb"]) + 0.35 * float(jugador_info.get("reb_l5", jugador_info["reb"]))
    ast_base = 0.65 * float(jugador_info["ast"]) + 0.35 * float(jugador_info.get("ast_l5", jugador_info["ast"]))
    tpm_base = 0.65 * float(jugador_info["tpm"]) + 0.35 * float(jugador_info.get("tpm_l5", jugador_info["tpm"]))

    mu_pts = max(2.0, pts_base * mult_ofensivo)
    mu_reb = max(1.0, reb_base * mult_rebotes)
    mu_ast = max(0.8, ast_base * mult_ofensivo)
    mu_tpm = max(0.1, tpm_base * mult_ofensivo)
    mu_pra = mu_pts + mu_reb + mu_ast

    min_base = float(jugador_info.get("min", 32.0))
    min_sim = np.clip(rng.normal(min_base, 3.4, size=n_sims), 14.0, 46.0)
    min_ratio = min_sim / max(15.0, min_base)

    sd_pts = float(jugador_info.get("sd_pts") or max(3.2, 0.27 * mu_pts))
    sd_reb = float(jugador_info.get("sd_reb") or max(1.6, 0.31 * mu_reb))
    sd_ast = float(jugador_info.get("sd_ast") or max(1.4, 0.33 * mu_ast))

    pts_sim = np.maximum(0.0, np.rint(rng.normal(mu_pts * min_ratio, sd_pts)))
    reb_sim = np.maximum(0.0, np.rint(rng.normal(mu_reb * min_ratio, sd_reb)))
    ast_sim = np.maximum(0.0, np.rint(rng.normal(mu_ast * min_ratio, sd_ast)))
    tpm_sim = rng.poisson(lam=np.clip(mu_tpm * min_ratio, 0.05, 12.0))
    pra_sim = pts_sim + reb_sim + ast_sim

    lineas_default = {
        "PTS": float(np.floor(mu_pts - 0.3) + 0.5),
        "REB": float(np.floor(mu_reb - 0.2) + 0.5),
        "AST": float(np.floor(mu_ast - 0.2) + 0.5),
        "3PM": float(max(0.5, np.floor(mu_tpm - 0.15) + 0.5)),
        "PRA": float(np.floor(mu_pra - 0.5) + 0.5),
    }
    if lineas_custom:
        for k, v in lineas_custom.items():
            if v is not None:
                lineas_default[k] = float(v)

    sims_map = {
        "PTS": (pts_sim, mu_pts, "Puntos (PTS)", float(jugador_info["pts"]), float(jugador_info.get("pts_l5", jugador_info["pts"]))),
        "REB": (reb_sim, mu_reb, "Rebotes (REB)", float(jugador_info["reb"]), float(jugador_info.get("reb_l5", jugador_info["reb"]))),
        "AST": (ast_sim, mu_ast, "Asistencias (AST)", float(jugador_info["ast"]), float(jugador_info.get("ast_l5", jugador_info["ast"]))),
        "3PM": (tpm_sim, mu_tpm, "Triples (3PM)", float(jugador_info["tpm"]), float(jugador_info.get("tpm_l5", jugador_info["tpm"]))),
        "PRA": (
            pra_sim,
            mu_pra,
            "Combo PRA (PTS+REB+AST)",
            float(jugador_info.get("pra", pts_base + reb_base + ast_base)),
            float(jugador_info.get("pra_l5", pts_base + reb_base + ast_base)),
        ),
    }

    mercados_out: Dict[str, Dict[str, Any]] = {}
    tabla_resumen: List[Dict[str, Any]] = []

    n_blocks = 10
    block_size = max(50, n_sims // n_blocks)

    for code, (arr_sim, media_esp, etiqueta, m_temp, m_l5) in sims_map.items():
        lin = lineas_default[code]
        arr_blocks = arr_sim[: n_blocks * block_size].reshape(n_blocks, block_size)
        over_b = np.mean(arr_blocks > lin, axis=1)
        under_b = 1.0 - over_b

        p_ov = float(np.clip(np.mean(over_b), 0.02, 0.98))
        p_un = float(np.clip(np.mean(under_b), 0.02, 0.98))
        p10_ov = float(np.clip(np.percentile(over_b, 10), 0.01, 0.98))
        p10_un = float(np.clip(np.percentile(under_b, 10), 0.01, 0.98))

        mejor_lado = "Over (Más de)" if p_ov >= p_un else "Under (Menos de)"
        p_mejor = p_ov if p_ov >= p_un else p_un
        p10_mejor = p10_ov if p_ov >= p_un else p10_un

        info_m = {
            "mercado": code,
            "etiqueta": etiqueta,
            "proyeccion": round(float(media_esp), 2),
            "linea": lin,
            "prob_over": round(p_ov, 4),
            "p10_over": round(p10_ov, 4),
            "cuota_justa_over": round(1.0 / p_ov, 2),
            "prob_under": round(p_un, 4),
            "p10_under": round(p10_un, 4),
            "cuota_justa_under": round(1.0 / p_un, 2),
            "recomendacion": f"{mejor_lado} {lin:.1f}",
            "prob_recomendacion": round(p_mejor, 4),
            "p10_recomendacion": round(p10_mejor, 4),
            "cuota_justa_recomendacion": round(1.0 / p_mejor, 2),
        }
        mercados_out[code] = info_m
        tabla_resumen.append(
            {
                "Mercado": etiqueta,
                "Media Temp.": round(m_temp, 1),
                "Racha L5": round(m_l5, 1),
                "Proyección Ajustada": round(float(media_esp), 2),
                "Línea": lin,
                "Over %": round(p_ov * 100.0, 1),
                "Cuota Over": round(1.0 / p_ov, 2),
                "Under %": round(p_un * 100.0, 1),
                "Cuota Under": round(1.0 / p_un, 2),
                "Pick Sugerido": f"{mejor_lado} {lin:.1f} ({p_mejor*100:.1f}%)",
            }
        )

    return {
        "jugador": jugador_info["jugador"],
        "equipo": jugador_info.get("equipo", ""),
        "pos": jugador_info.get("pos", ""),
        "estado_lesion": jugador_info.get("estado_lesion", "Activa"),
        "detalle_lesion": jugador_info.get("detalle_lesion", ""),
        "fuente": jugador_info.get("fuente", "Catálogo Base"),
        "min_esp": round(min_base * (0.96 if en_b2b else 1.0), 1),
        "mercados": mercados_out,
        "tabla_resumen": tabla_resumen,
    }


def proyectar_plantilla_partido(
    equipo_local: str,
    equipo_visita: str,
    pace_esperado: float = NBA_AVG_PACE,
    drtg_local: float = NBA_AVG_ORTG,
    drtg_visita: float = NBA_AVG_ORTG,
    b2b_local: bool = False,
    b2b_visita: bool = False,
    bajas_local: Optional[List[str]] = None,
    bajas_visita: Optional[List[str]] = None,
    n_sims: int = 3500,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Genera la tabla proyectada de todos los jugadores activos de ambos equipos en un partido,
    redistribuyendo automáticamente el Usage% y tiros cuando hay bajas seleccionadas o detectadas.
    """
    canon_l = normalizar_equipo(equipo_local)
    canon_v = normalizar_equipo(equipo_visita)
    bajas_l_set = set(bajas_local or [])
    bajas_v_set = set(bajas_visita or [])

    roster_l = obtener_jugadores_equipo(canon_l, db_path=db_path)
    roster_v = obtener_jugadores_equipo(canon_v, db_path=db_path)

    usg_liberado_l = sum(float(p["usg"]) for p in roster_l if p["jugador"] in bajas_l_set)
    usg_liberado_v = sum(float(p["usg"]) for p in roster_v if p["jugador"] in bajas_v_set)

    proyecciones_l: List[Dict[str, Any]] = []
    proyecciones_v: List[Dict[str, Any]] = []
    mejores_props_partido: List[Dict[str, Any]] = []

    for idx, p in enumerate(roster_l):
        if p["jugador"] in bajas_l_set:
            continue
        sim_p = proyectar_y_simular_jugador(
            p,
            es_local=True,
            pace_esperado=pace_esperado,
            drtg_rival=drtg_visita,
            en_b2b=b2b_local,
            bajas_companeros_usg=usg_liberado_l,
            n_sims=n_sims,
            seed=100 + idx,
        )
        proyecciones_l.append(sim_p)
        for m_code, m_data in sim_p["mercados"].items():
            mejores_props_partido.append(
                {
                    "Jugador": p["jugador"],
                    "Equipo": canon_l,
                    "Estado": sim_p["estado_lesion"],
                    "Mercado": m_data["etiqueta"],
                    "Proyección": m_data["proyeccion"],
                    "Selección": m_data["recomendacion"],
                    "Probabilidad %": round(m_data["prob_recomendacion"] * 100.0, 1),
                    "Piso P10 %": round(m_data["p10_recomendacion"] * 100.0, 1),
                    "Cuota Justa": m_data["cuota_justa_recomendacion"],
                }
            )

    for idx, p in enumerate(roster_v):
        if p["jugador"] in bajas_v_set:
            continue
        sim_p = proyectar_y_simular_jugador(
            p,
            es_local=False,
            pace_esperado=pace_esperado,
            drtg_rival=drtg_local,
            en_b2b=b2b_visita,
            bajas_companeros_usg=usg_liberado_v,
            n_sims=n_sims,
            seed=200 + idx,
        )
        proyecciones_v.append(sim_p)
        for m_code, m_data in sim_p["mercados"].items():
            mejores_props_partido.append(
                {
                    "Jugador": p["jugador"],
                    "Equipo": canon_v,
                    "Estado": sim_p["estado_lesion"],
                    "Mercado": m_data["etiqueta"],
                    "Proyección": m_data["proyeccion"],
                    "Selección": m_data["recomendacion"],
                    "Probabilidad %": round(m_data["prob_recomendacion"] * 100.0, 1),
                    "Piso P10 %": round(m_data["p10_recomendacion"] * 100.0, 1),
                    "Cuota Justa": m_data["cuota_justa_recomendacion"],
                }
            )

    mejores_props_partido.sort(key=lambda x: (x["Piso P10 %"], x["Probabilidad %"]), reverse=True)

    return {
        "local": canon_l,
        "visita": canon_v,
        "jugadores_local": proyecciones_l,
        "jugadores_visita": proyecciones_v,
        "top_props": mejores_props_partido[:8],
    }
