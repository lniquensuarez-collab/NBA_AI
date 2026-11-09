"""
Suite de Pruebas Unitarias y Cuantitativas para NBA ULTIMATE AI (`tests/test_engine.py`).
Ejecuta validaciones exhaustivas sin conexión de red utilizando datos sintéticos NBA:
1. Catálogo de las 30 franquicias NBA, normalización y alturas de pabellones (Denver/Utah).
2. Esquema SQLite, migraciones, inserción sin duplicados, fórmula de posesiones de Dean Oliver,
   normalización de prórrogas (OT) a 48 min y desmargenado por Método de Potencia (Power Method).
3. Importador de archivos CSV/Excel del usuario (`importar_archivo_usuario`).
4. Sistema Elo NBA (MOV FiveThirtyEight, spread_elo = diff_elo / 28.0 y regresión 25% entre temporadas).
5. Cálculo cronológico de Features sin fuga de información (Zero Data Leakage), Four Factors EWMA 70/30,
   detector de fatiga B2B (-2.6 visita, -1.5 local, -1.2 por 3-in-4, +1.0 por Rest Advantage) y altitud (+40% con fatiga).
6. Motor Ensemble de 3 Pilares, Simulación de Monte Carlo correlacionada, Bono de Pago Anticipado +20 pts,
   Semáforo de Consenso (`diagnosticar_consenso`), Selector de Bajo Riesgo (regla >= 70% ML vs Hándicap)
   y Armador de Combinadas + Zona Picante (Cuota >= 5.00).
7. Backtest Walk-Forward cronológico y simulación financiera de banca.
"""

import io
import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from engine.config import (
    NBA_TEAMS,
    ALTITUD_NBA,
    normalizar_equipo,
)
from engine.db import (
    init_db,
    guardar_partidos,
    cargar_partidos_df,
    calcular_posesiones_dean_oliver,
    desmargenar_cuotas_power,
)
from engine.data_bot import (
    generar_partidos_sinteticos_nba,
    importar_archivo_usuario,
)
from engine.elo_system import (
    EloSystemNBA,
    calcular_multiplicador_mov,
    probabilidad_elo,
)
from engine.features import (
    ajuste_fatiga_nba,
    calcular_ajuste_altitud,
    construir_dataset_cronologico,
    generar_tabla_power_ranking,
)
from engine.models import (
    NBAEnsembleEngine,
    beneficio_pago_anticipado_20pts,
    diagnosticar_consenso,
    calcular_ev_y_kelly,
    selecciones_bajo_riesgo,
    armar_combinadas_y_zona_picante,
)
from engine.backtest import ejecutar_backtest_walk_forward
from engine.players import (
    obtener_jugadores_equipo,
    obtener_bajas_detectadas_equipo,
    obtener_catalogo_completo_jugadores,
    guardar_jugadores_db,
    calcular_impacto_bajas,
    proyectar_y_simular_jugador,
    proyectar_plantilla_partido,
)


class TestNBAUltimateEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_nba.db")
        init_db(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_01_catalogo_30_franquicias_y_altitud(self) -> None:
        self.assertEqual(len(NBA_TEAMS), 30, "Deben existir exactamente las 30 franquicias NBA.")
        self.assertEqual(normalizar_equipo("Lakers"), "Los Angeles Lakers")
        self.assertEqual(normalizar_equipo("BOS"), "Boston Celtics")
        self.assertEqual(normalizar_equipo("Sixers"), "Philadelphia 76ers")
        self.assertEqual(ALTITUD_NBA["Denver Nuggets"]["altitud_msnm"], 1609)
        self.assertEqual(ALTITUD_NBA["Utah Jazz"]["altitud_msnm"], 1288)
        print("✅ [1/7] Catálogo de 30 franquicias NBA y metadatos de altitud verificados.")

    def test_02_sqlite_dean_oliver_y_power_method(self) -> None:
        poss = calcular_posesiones_dean_oliver(fga=88.0, tov=13.0, fta=22.0, oreb=10.0)
        expected_poss = 0.96 * (88.0 + 13.0 + 0.44 * 22.0 - 10.0)
        self.assertAlmostEqual(poss, expected_poss, places=4)

        partidos = generar_partidos_sinteticos_nba(n_partidos=80, temporadas=[2025, 2026], db_path=self.db_path)
        # Re-insertar los mismos partidos para verificar que no se dupliquen
        guardar_partidos(partidos[:20], db_path=self.db_path)
        df = cargar_partidos_df(db_path=self.db_path)
        self.assertEqual(len(df), len(partidos))
        for col in ["poss", "ortg_l", "drtg_l", "ortg_v", "drtg_v", "q1_l", "h1_l", "fue_ot"]:
            self.assertIn(col, df.columns)
            self.assertFalse(df[col].isna().any())

        # Desmargenado Power Method
        p_l, p_v, overround = desmargenar_cuotas_power(1.91, 1.91)
        self.assertAlmostEqual(p_l, 0.5, places=4)
        self.assertAlmostEqual(p_v, 0.5, places=4)
        self.assertGreater(overround, 4.0)
        print("✅ [2/7] Esquema SQLite, Posesiones Dean Oliver y Power Method verificados.")

    def test_03_importador_csv_usuario(self) -> None:
        csv_content = (
            "fecha,local,visita,pts_l,pts_v,cuota_l,cuota_v\n"
            "2026-01-15,Boston Celtics,Miami Heat,118,108,1.35,3.25\n"
            "2026-01-16,Denver Nuggets,Utah Jazz,122,114,1.28,3.80\n"
        )
        buf = io.StringIO(csv_content)
        res = importar_archivo_usuario(buf, nombre_archivo="partidos_usuario.csv", db_path=self.db_path)
        self.assertEqual(res["insertados"], 2)
        df = cargar_partidos_df(db_path=self.db_path)
        self.assertEqual(len(df), 2)
        print("✅ [3/7] Importador de archivos CSV/Excel del usuario verificado.")

    def test_04_sistema_elo_mov_y_regresion(self) -> None:
        mult = calcular_multiplicador_mov(pts_l=115, pts_v=100, elo_diff_ganador=50.0)
        expected_mult = ((15 + 3) ** 0.8) / (7.5 + 0.006 * 50.0)
        self.assertAlmostEqual(mult, expected_mult, places=5)

        p_elo, spread_elo, diff_elo = probabilidad_elo(1556.0, 1500.0, hca_elo=0.0)
        self.assertAlmostEqual(spread_elo, 56.0 / 28.0, places=5)
        self.assertGreater(p_elo, 0.5)

        elo_sys = EloSystemNBA()
        elo_sys.actualizar_partido("Boston Celtics", "Detroit Pistons", 125, 102, temporada=2025, fecha="2025-02-01")
        elo_bos_2025 = elo_sys.get_rating("Boston Celtics")
        self.assertGreater(elo_bos_2025, 1500.0)

        # Al cambiar a temporada 2026 debe regresar un 25% hacia 1500
        elo_sys.actualizar_partido("Boston Celtics", "Detroit Pistons", 110, 105, temporada=2026, fecha="2025-10-25")
        expected_pre_2026 = 0.75 * elo_bos_2025 + 0.25 * 1500.0
        self.assertAlmostEqual(elo_sys.history[-1]["elo_l_pre"], expected_pre_2026, places=4)
        print("✅ [4/7] Sistema Elo NBA (MOV, spread_elo = diff/28 y regresión 25%) verificado.")

    def test_05_fatiga_calendario_y_altitud_nba(self) -> None:
        # Caso 1: Visitante en B2B (-2.6 pts) y además 3-in-4 (-1.2 pts) jugando en Denver (+1.6 * 1.4 = +2.24 pts)
        # mientras Denver descansó 2 días:
        fechas_den = ["2026-01-07"]
        fechas_lal = ["2026-01-07", "2026-01-09"]  # Juega el 2026-01-10 -> B2B (0d descanso) y 3er partido en 4 noches
        fat = ajuste_fatiga_nba(
            local="Denver Nuggets",
            visita="Los Angeles Lakers",
            fechas_local=fechas_den,
            fechas_visita=fechas_lal,
            fecha_partido="2026-01-10",
        )
        self.assertTrue(fat["b2b_v"])
        self.assertTrue(fat["densidad_v"])
        self.assertAlmostEqual(fat["ajuste_fatiga_v_pts"], -2.6 + (-1.2), places=2)
        self.assertAlmostEqual(fat["bono_altitud_l_pts"], round(1.6 * 1.40, 2), places=2)
        self.assertIn("Alerta de Fatiga", fat["tarjeta_diagnostico"])

        # Caso 2: Neutralización mutua Denver vs Utah
        alt_neutral = calcular_ajuste_altitud("Denver Nuggets", "Utah Jazz", visita_en_fatiga=True)
        self.assertTrue(alt_neutral["neutralizado"])
        self.assertEqual(alt_neutral["bono_local_pts"], 0.0)

        # Caso 3: Rest Advantage (>= 3 días vs 1 día -> +1.0 pt)
        fat_rest = ajuste_fatiga_nba(
            local="Miami Heat",
            visita="Chicago Bulls",
            fechas_local=["2026-01-05"],   # 4 días de descanso antes del 2026-01-10
            fechas_visita=["2026-01-08"],  # 1 día de descanso antes del 2026-01-10
            fecha_partido="2026-01-10",
        )
        self.assertAlmostEqual(fat_rest["ajuste_fatiga_l_pts"], 1.0, places=2)
        print("✅ [5/7] Detector de Fatiga B2B, 3-in-4, Rest Advantage y Altitud NBA verificados.")

    def test_06_ensemble_montecarlo_pago_anticipado_y_combinadas(self) -> None:
        generar_partidos_sinteticos_nba(n_partidos=300, temporadas=[2026], db_path=self.db_path)
        df = cargar_partidos_df(db_path=self.db_path)
        df_feat, tracker, elo_sys = construir_dataset_cronologico(df)

        engine = NBAEnsembleEngine(half_life=120.0, n_sims=5000)
        engine.tracker = tracker
        engine.elo_sys = elo_sys
        engine.fit(df, df_features_precomputed=df_feat)

        pred = engine.predecir_partido_completo(
            local="Boston Celtics",
            visita="Charlotte Hornets",
            fecha="2026-03-20",
            casa_apuestas="Betano (Pago Anticipado +20 pts / SuperCuotas)",
            n_sims=5000,
        )

        # Verificar Monte Carlo y mercados
        mc = pred["monte_carlo"]
        self.assertEqual(mc["n_sims"], 5000)
        self.assertAlmostEqual(mc["moneyline"]["prob_local"] + mc["moneyline"]["prob_visita"], 1.0, places=3)
        self.assertGreaterEqual(len(mc["spread"]["tabla"]), 20)
        self.assertEqual(len(mc["totales"]["tabla"]), 7)
        self.assertIn("q1", mc)
        self.assertIn("h1", mc)

        # Verificar Pago Anticipado +20 pts
        pa = pred["pago_anticipado"]
        self.assertGreater(pa["p_efectiva_local"], pa["p_ml_local_base"])
        self.assertLess(pa["cuota_justa_pa_local"], pa["cuota_justa_base_local"])

        # Verificar criterio de Bajo Riesgo cuando hay favorito >= 70%
        pred_fuerte = dict(pred)
        pred_fuerte["monte_carlo"] = dict(mc)
        pred_fuerte["monte_carlo"]["moneyline"] = dict(mc["moneyline"], prob_local=0.75, prob_visita=0.25, p10_local=0.70)
        pred_fuerte["pago_anticipado"] = beneficio_pago_anticipado_20pts(118.0, 108.0, 11.5, 0.75)
        picks_fuerte = selecciones_bajo_riesgo(pred_fuerte)
        self.assertIn("Moneyline", picks_fuerte[0]["mercado"])

        # Verificar Semáforo de Consenso (Partido Trampa si discrepancia >= 12 pp)
        cons_trampa = diagnosticar_consenso(p_eff=0.68, p_elo=0.51, p_ml_clf=0.58, min_pj_season=20)
        self.assertTrue(cons_trampa["es_partido_trampa"])

        # Verificar Armador de Combinadas y Zona Picante (Cuota >= 5.00)
        jornada = [
            engine.predecir_partido_completo("Boston Celtics", "Indiana Pacers", "2026-03-20", n_sims=5000, seed=1),
            engine.predecir_partido_completo("Denver Nuggets", "Phoenix Suns", "2026-03-20", n_sims=5000, seed=2),
            engine.predecir_partido_completo("Oklahoma City Thunder", "Dallas Mavericks", "2026-03-20", n_sims=5000, seed=3),
            engine.predecir_partido_completo("Cleveland Cavaliers", "Milwaukee Bucks", "2026-03-20", n_sims=5000, seed=4),
            engine.predecir_partido_completo("New York Knicks", "Golden State Warriors", "2026-03-20", n_sims=5000, seed=5),
        ]
        combos = armar_combinadas_y_zona_picante(jornada)
        self.assertIsNotNone(combos["mejor_doble"])
        self.assertIsNotNone(combos["mejor_triple"])
        esc = combos["zona_picante"]["escalera_picante"]
        bomba = combos["zona_picante"]["bomba_express"]
        self.assertIsNotNone(esc)
        self.assertIsNotNone(bomba)
        self.assertGreaterEqual(esc["cuota_combinada"], 5.00)
        self.assertGreaterEqual(bomba["cuota_combinada"], 5.00)

        # Verificar Power Ranking
        df_rank = generar_tabla_power_ranking(tracker, elo_sys)
        self.assertEqual(len(df_rank), 30)
        print("✅ [6/7] Ensemble 3 Pilares, Monte Carlo, Pago Anticipado +20 pts, Bajo Riesgo y Zona Picante 5.0+ verificados.")

    def test_07_backtest_walk_forward(self) -> None:
        generar_partidos_sinteticos_nba(n_partidos=110, temporadas=[2025, 2026], db_path=self.db_path)
        df = cargar_partidos_df(db_path=self.db_path)
        bt = ejecutar_backtest_walk_forward(df, warmup_games=35, retrain_step=25)
        self.assertEqual(bt["status"], "ok")
        self.assertEqual(len(bt["tabla_comparativa"]), 3)
        self.assertIn("log_loss", bt["metricas_ensemble"])
        self.assertIn("mae_spread_pts", bt["metricas_ensemble"])
        self.assertIn("roi_yield_pct", bt["simulacion_financiera"])
        print("✅ [7/8] Backtest Walk-Forward cronológico y Simulación Financiera verificados.")

    def test_08_estadisticas_jugadores_props_y_bajas(self) -> None:
        df_cat = obtener_catalogo_completo_jugadores(db_path=self.db_path)
        self.assertGreaterEqual(len(df_cat), 90)
        self.assertIn("PRA (PTS+REB+AST)", df_cat.columns)
        self.assertIn("PTS (L5)", df_cat.columns)
        self.assertIn("Estado", df_cat.columns)

        bos_players = obtener_jugadores_equipo("Boston Celtics", db_path=self.db_path)
        self.assertGreaterEqual(len(bos_players), 4)
        tatum = bos_players[0]
        sim_tatum = proyectar_y_simular_jugador(tatum, es_local=True, pace_esperado=100.5, drtg_rival=115.2, n_sims=3000)
        for m_code in ["PTS", "REB", "AST", "3PM", "PRA"]:
            self.assertIn(m_code, sim_tatum["mercados"])
            self.assertAlmostEqual(
                sim_tatum["mercados"][m_code]["prob_over"] + sim_tatum["mercados"][m_code]["prob_under"],
                1.0,
                places=3,
            )

        # Verificar persistencia SQLite de estado de lesión y racha L5
        tatum_lesionado = dict(tatum, estado_lesion="Baja (Out)", pts_l5=32.4)
        guardar_jugadores_db([tatum_lesionado], db_path=self.db_path)
        bajas_auto = obtener_bajas_detectadas_equipo("Boston Celtics", db_path=self.db_path)
        self.assertIn(tatum["jugador"], bajas_auto)

        bajas_info = calcular_impacto_bajas(
            "Boston Celtics", "Los Angeles Lakers", bajas_local=["Jayson Tatum"], bajas_visita=[], db_path=self.db_path
        )
        self.assertLess(bajas_info["ajuste_spread_local_pts"], 0.0)
        self.assertLess(bajas_info["ajuste_total_pts"], 0.0)

        plantilla = proyectar_plantilla_partido("Boston Celtics", "Los Angeles Lakers", n_sims=2000, db_path=self.db_path)
        self.assertGreaterEqual(len(plantilla["top_props"]), 5)
        print("✅ [8/8] Estadísticas de Jugadores SQLite/ESPN, Racha L5, Player Props y Simulador de Bajas verificados.")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestNBAUltimateEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
