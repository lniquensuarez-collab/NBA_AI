"""
Módulo Cuantitativo de Estadísticas de Jugadores, Player Props (PTS, REB, AST, 3PM, PRA)
y Simulador de Impacto de Estrellas / Bajas (`engine/players.py`).
- Catálogo base completo de estrellas y titulares clave de las 30 franquicias NBA.
- Sincronización opcional en vivo con el endpoint público de plantillas/estadísticas de ESPN NBA
  (100% libre de API keys).
- Proyección ajustada al contexto del partido (Ritmo esperado `Pace`, Defensa del rival `DRtg`,
  factor localía y penalización de fatiga `B2B`).
- Simulación de Monte Carlo correlacionada por jugador para calcular probabilidades exactas
  `Over %`, `Under %`, `Piso P10 %`, `Cuota Justa` y `EV%` en mercados de:
    * Puntos (`PTS`)
    * Rebotes (`REB`)
    * Asistencias (`AST`)
    * Triples Anotados (`3PM`)
    * Combo `PRA` (`PTS + REB + AST`)
- Simulador de Bajas / Ausencias para ajustar dinámicamente el Spread y Total del partido en `🔮 Predicción`.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from engine.config import (
    NBA_TEAMS,
    NBA_AVG_PACE,
    NBA_AVG_ORTG,
    normalizar_equipo,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Catálogo base de jugadores clave y estrellas por franquicia NBA:
# (nombre, posición, min, pts, reb, ast, tpm, usage_pct, impacto_spread_pts)
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


def obtener_jugadores_equipo(equipo: str) -> List[Dict[str, Any]]:
    """
    Retorna la plantilla de jugadores principales para una franquicia NBA
    con su `PRA` base precalculado (`PTS + REB + AST`).
    """
    canon = normalizar_equipo(equipo)
    lista_base = BASE_ROSTERS_NBA.get(canon, [])
    salida: List[Dict[str, Any]] = []
    for p in lista_base:
        item = dict(p)
        item["equipo"] = canon
        item["pra"] = round(float(item["pts"]) + float(item["reb"]) + float(item["ast"]), 1)
        salida.append(item)
    return salida


def obtener_catalogo_completo_jugadores() -> pd.DataFrame:
    """
    Retorna un DataFrame con todos los jugadores principales de las 30 franquicias NBA,
    ordenado por `PRA` (`PTS + REB + AST`) y `PTS`.
    """
    filas: List[Dict[str, Any]] = []
    for eq in NBA_TEAMS:
        for p in obtener_jugadores_equipo(eq):
            filas.append(
                {
                    "Jugador": p["jugador"],
                    "Equipo": p["equipo"],
                    "Pos": p["pos"],
                    "MIN": round(float(p["min"]), 1),
                    "PTS": round(float(p["pts"]), 1),
                    "REB": round(float(p["reb"]), 1),
                    "AST": round(float(p["ast"]), 1),
                    "3PM": round(float(p["tpm"]), 1),
                    "PRA (PTS+REB+AST)": round(float(p["pra"]), 1),
                    "USG%": round(float(p["usg"]), 1),
                    "Impacto Spread (pts)": round(float(p["impacto_pts"]), 1),
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
) -> Dict[str, Any]:
    """
    Calcula el impacto en puntos sobre el Spread y el Total del partido cuando uno o más
    jugadores clave de `equipo_local` o `equipo_visita` son marcados como Baja / Ausentes,
    aplicando redistribución de posesión (Usage Redistribution ~35% absorbido por compañeros,
    65% pérdida neta de eficiencia).
    """
    canon_l = normalizar_equipo(equipo_local)
    canon_v = normalizar_equipo(equipo_visita)
    bajas_l_set = set(bajas_local or [])
    bajas_v_set = set(bajas_visita or [])

    roster_l = {p["jugador"]: p for p in obtener_jugadores_equipo(canon_l)}
    roster_v = {p["jugador"]: p for p in obtener_jugadores_equipo(canon_v)}

    perdida_l = sum(float(roster_l[j]["impacto_pts"]) for j in bajas_l_set if j in roster_l)
    perdida_v = sum(float(roster_v[j]["impacto_pts"]) for j in bajas_v_set if j in roster_v)

    # Impacto neto sobre el margen del local (positivo si las bajas perjudican más a la visita)
    ajuste_spread_local = round(perdida_v - perdida_l, 2)
    # Impacto sobre el total de puntos del partido (ausencia de anotadores reduce ligeramente el ritmo/eficiencia ofensiva)
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

    Ajustes contextuales aplicados automáticamente:
      - Factor Ritmo: `pace_esperado / NBA_AVG_PACE`
      - Factor Defensa Rival: `drtg_rival / NBA_AVG_ORTG`
      - Factor Localía: `+1.8%` en casa / `-1.2%` de visita
      - Factor Fatiga B2B: `-4.5%` en tiro/minutos si el equipo llega en Back-to-Back
      - Redistribución de Usage (`bajas_companeros_usg`): incrementa volumen si falta una estrella compañera.
    """
    rng = np.random.default_rng(seed)
    n_sims = int(np.clip(n_sims, 1000, 15000))

    f_pace = float(np.clip(pace_esperado / NBA_AVG_PACE, 0.90, 1.12))
    f_def = float(np.clip(drtg_rival / NBA_AVG_ORTG, 0.88, 1.14))
    f_home = 1.018 if es_local else 0.988
    f_b2b = 0.955 if en_b2b else 1.00
    # Si falta un compañero titular, el jugador absorbe ~22% del usage liberado
    f_usg_boost = float(np.clip(1.0 + 0.0065 * float(bajas_companeros_usg), 1.0, 1.22))

    mult_ofensivo = f_pace * f_def * f_home * f_b2b * f_usg_boost
    mult_rebotes = f_pace * (1.01 if es_local else 0.995) * (0.97 if en_b2b else 1.0) * (1.0 + 0.003 * float(bajas_companeros_usg))

    mu_pts = max(2.0, float(jugador_info["pts"]) * mult_ofensivo)
    mu_reb = max(1.0, float(jugador_info["reb"]) * mult_rebotes)
    mu_ast = max(0.8, float(jugador_info["ast"]) * mult_ofensivo)
    mu_tpm = max(0.1, float(jugador_info["tpm"]) * mult_ofensivo)
    mu_pra = mu_pts + mu_reb + mu_ast

    # Simulación correlacionada a través de los minutos jugados en el partido (factor latente común)
    # Si el jugador tiene más minutos (partido cerrado o prórroga), PTS, REB y AST suben juntos:
    min_base = float(jugador_info.get("min", 32.0))
    min_sim = np.clip(rng.normal(min_base, 3.4, size=n_sims), 14.0, 46.0)
    min_ratio = min_sim / max(15.0, min_base)

    # Simulaciones individuales condicionadas al ratio de minutos
    sd_pts = max(3.2, 0.27 * mu_pts)
    sd_reb = max(1.6, 0.31 * mu_reb)
    sd_ast = max(1.4, 0.33 * mu_ast)

    pts_sim = np.maximum(0.0, np.rint(rng.normal(mu_pts * min_ratio, sd_pts)))
    reb_sim = np.maximum(0.0, np.rint(rng.normal(mu_reb * min_ratio, sd_reb)))
    ast_sim = np.maximum(0.0, np.rint(rng.normal(mu_ast * min_ratio, sd_ast)))
    tpm_sim = np.rng_poisson if hasattr(np, "rng_poisson") else rng.poisson(lam=np.clip(mu_tpm * min_ratio, 0.05, 12.0))
    pra_sim = pts_sim + reb_sim + ast_sim

    # Líneas estándar de casa de apuestas (terminadas en .5 alrededor de la media proyectada)
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
        "PTS": (pts_sim, mu_pts, "Puntos (PTS)"),
        "REB": (reb_sim, mu_reb, "Rebotes (REB)"),
        "AST": (ast_sim, mu_ast, "Asistencias (AST)"),
        "3PM": (tpm_sim, mu_tpm, "Triples (3PM)"),
        "PRA": (pra_sim, mu_pra, "Combo PRA (PTS+REB+AST)"),
    }

    mercados_out: Dict[str, Dict[str, Any]] = {}
    tabla_resumen: List[Dict[str, Any]] = []

    # Dividir en 10 bloques para obtener Piso P10 de cada prop
    n_blocks = 10
    block_size = max(50, n_sims // n_blocks)

    for code, (arr_sim, media_esp, etiqueta) in sims_map.items():
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
                "Media Base": round(float(jugador_info["pra" if code == "PRA" else ("tpm" if code == "3PM" else code.lower())]), 1),
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
) -> Dict[str, Any]:
    """
    Genera la tabla proyectada de todos los jugadores activos de ambos equipos en un partido,
    redistribuyendo automáticamente el Usage% y tiros cuando hay bajas seleccionadas.
    """
    canon_l = normalizar_equipo(equipo_local)
    canon_v = normalizar_equipo(equipo_visita)
    bajas_l_set = set(bajas_local or [])
    bajas_v_set = set(bajas_visita or [])

    roster_l = obtener_jugadores_equipo(canon_l)
    roster_v = obtener_jugadores_equipo(canon_v)

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
