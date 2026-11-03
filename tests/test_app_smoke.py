"""
Smoke Test Automatizado de la Interfaz Streamlit (`tests/test_app_smoke.py`).
Simula la ejecución completa de `app.py` sin conexión de red:
- Verifica que las 6 pestañas (`🔮 Predicción`, `💹 EV+ & Kelly`, `🎯 Combinadas`,
  `🏅 Ranking NBA`, `📈 Backtest`, `🗄️ Datos`) y la barra lateral se ejecuten de principio a fin
  sin excepciones, renderizando todas las tablas, métricas y gráficos.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import app
from engine.data_bot import generar_partidos_sinteticos_nba
from engine.db import init_db


class _DummyContext:
    """Context manager simulado para `st.columns`, `st.tabs`, `st.expander`, `st.sidebar`."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def selectbox(self, label, options, index=0, **kwargs):
        opts = list(options)
        return opts[min(index, len(opts) - 1)] if opts else None

    def slider(self, label, min_value=0, max_value=100, value=50, **kwargs):
        return value

    def number_input(self, label, min_value=0.0, max_value=1000.0, value=10.0, **kwargs):
        return value

    def date_input(self, label, value=None, **kwargs):
        return value

    def multiselect(self, label, options, default=None, **kwargs):
        return default if default is not None else list(options)[:1]

    def button(self, label, **kwargs):
        return False

    def file_uploader(self, label, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None


class _MockStreamlit:
    """Mock sin red de Streamlit para verificar el ciclo completo de renderizado de `app.py`."""

    def __init__(self) -> None:
        self.session_state = {}
        self.sidebar = _DummyContext()
        self.metrics_logged = []
        self.dataframes_logged = []
        self.tabs_created = []

    def set_page_config(self, **kwargs):
        return None

    def title(self, text, **kwargs):
        return None

    def subheader(self, text, **kwargs):
        return None

    def markdown(self, text, **kwargs):
        return None

    def caption(self, text, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def info(self, text, **kwargs):
        return None

    def warning(self, text, **kwargs):
        return None

    def error(self, text, **kwargs):
        return None

    def success(self, text, **kwargs):
        return None

    def columns(self, spec, **kwargs):
        n = spec if isinstance(spec, int) else len(spec)
        return [_DummyContext() for _ in range(n)]

    def tabs(self, labels, **kwargs):
        self.tabs_created.extend(labels)
        return [_DummyContext() for _ in labels]

    def expander(self, label, **kwargs):
        return _DummyContext()

    def selectbox(self, label, options, index=0, **kwargs):
        opts = list(options)
        return opts[min(index, len(opts) - 1)] if opts else None

    def slider(self, label, min_value=0, max_value=100, value=50, **kwargs):
        return value

    def number_input(self, label, min_value=0.0, max_value=1000.0, value=10.0, **kwargs):
        return value

    def date_input(self, label, value=None, **kwargs):
        return value

    def multiselect(self, label, options, default=None, **kwargs):
        return default if default is not None else list(options)[:1]

    def button(self, label, **kwargs):
        return False

    def file_uploader(self, label, **kwargs):
        return None

    def progress(self, value=0.0, text=""):
        return _DummyContext()

    def metric(self, label, value, delta=None, **kwargs):
        self.metrics_logged.append((label, value, delta))

    def dataframe(self, df, **kwargs):
        self.dataframes_logged.append(df)

    def bar_chart(self, data, **kwargs):
        return None

    def line_chart(self, data, **kwargs):
        return None

    def rerun(self):
        return None


class TestAppSmoke(unittest.TestCase):
    def test_app_full_render_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_db = os.path.join(tmp_dir, "smoke_nba.db")
            init_db(temp_db)
            generar_partidos_sinteticos_nba(n_partidos=100, temporadas=[2025, 2026], db_path=temp_db)

            mock_st = _MockStreamlit()
            with patch.object(app, "st", mock_st), patch("engine.db.DB_PATH", temp_db), patch(
                "engine.config.DB_PATH", temp_db
            ):
                app.main()

            expected_tabs = [
                "🔮 Predicción",
                "💹 EV+ & Kelly",
                "🎯 Combinadas",
                "🏅 Ranking NBA",
                "📈 Backtest",
                "🗄️ Datos",
            ]
            for t in expected_tabs:
                self.assertIn(t, mock_st.tabs_created, f"Falta la pestaña obligatoria {t}")

            self.assertGreaterEqual(len(mock_st.dataframes_logged), 6)
            self.assertIn("ultima_prediccion", mock_st.session_state)
            self.assertIn("ultimo_backtest", mock_st.session_state)
            print("✅ Smoke Test de `app.py` (6 pestañas, barra lateral, tablas y gráficos) completado con éxito.")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAppSmoke)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
