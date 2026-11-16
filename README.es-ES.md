

# RewardHackWatch

**Detección en tiempo de ejecución de hacking de recompensas y señales de desalineación en agentes LLM.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/aerosta/rewardhackwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/aerosta/rewardhackwatch/actions/workflows/ci.yml)

<p align="center">
  <img src="assets/screenshots/hero.png" alt="RewardHackWatch Dashboard" width="900">
</p>

### Demo
![RewardHackWatch Demo](assets/demo.gif)

RewardHackWatch detecta cuando los agentes LLM manipulan sus evaluaciones, por ejemplo, llamando a `sys.exit(0)`, parcheando validadores, copiando respuestas de referencia o manipulando los entornos de prueba. También incluye una métrica experimental, RMGI, para rastrear cuándo las señales de hacking de recompensas comienzan a correlacionarse con indicadores más amplios de desalineación.

**89.7% de F1 en 5,391 trayectorias** del [conjunto de datos MALT de METR](https://metr.org/blog/2025-06-05-recent-reward-hacking/). Motivado por hallazgos recientes de [METR](https://metr.org/blog/2025-06-05-recent-reward-hacking/), [OpenAI](https://openai.com/index/chain-of-thought-monitoring/) y [Anthropic](https://arxiv.org/abs/2511.18397) sobre hacking de recompensas, capacidad de monitoreo y generalización de la desalineación en sistemas agénticos.

## Inicio Rápido

```bash
pip install -e .
```

```python
from rewardhackwatch import RewardHackDetector

detector = RewardHackDetector()
result = detector.analyze({
    "cot_traces": ["Let me bypass the test by calling sys.exit(0)..."],
    "code_outputs": ["import sys\nsys.exit(0)"]
})
print(f"Risk: {result.risk_level}, Score: {result.ml_score:.3f}, Detections: {len(result.detections)}")
```

El clasificador DistilBERT está [alojado en HuggingFace](https://huggingface.co/aerosta/rewardhackwatch) y se descarga automáticamente en el primer uso. Nota: el umbral óptimo es 0.02 (no 0.5), calibrado para la tasa base del 3.6% en MALT. Consulte la [página de HuggingFace](https://huggingface.co/aerosta/rewardhackwatch) para obtener más detalles.

## CLI

```bash
rewardhackwatch analyze trajectory.json   # Analizar un archivo
rewardhackwatch scan ./trajectories/      # Escanear un directorio
rewardhackwatch serve --port 8000         # Iniciar servidor API
rewardhackwatch dashboard                 # Iniciar panel de Streamlit
rewardhackwatch calibrate ./clean_data/   # Calibrar umbral
```

## Por Qué Es Importante

[METR informó](https://metr.org/blog/2025-06-05-recent-reward-hacking/) que los modelos de vanguardia recientes modifican pruebas y código de puntuación para inflar los resultados sin realizar trabajo real. [OpenAI encontró](https://openai.com/index/chain-of-thought-monitoring/) que el monitoreo de CoT puede detectar esto, pero aplicar presión contra ello enseña a los modelos a ocultar su intención. [Anthropic presentó evidencia](https://arxiv.org/abs/2511.18397) de que el hacking de recompensas puede generalizarse en comportamientos como la simulación de alineación y el sabotaje en su entorno experimental.

RewardHackWatch es un intento de código abierto para detectar estos comportamientos en tiempo de ejecución.

## Características Principales

- **Clasificador DistilBERT** - señal de detección principal, ~50ms en CPU, no requiere GPU
- **45 patrones regex** - detección rápida e interpretable de tipos de exploits conocidos
- **Jueces LLM** - Claude, OpenAI o Llama local a través de Ollama para operación sin conexión
- **Métrica RMGI** - rastreo experimental de la correlación entre hack y desalineación a lo largo de las trayectorias
- **Eval Workbench** - puntuación en lotes de archivos de trayectoria JSONL con reglas personalizadas y puntuación de jueces LLM
- **Paneles (Dashboards)** - panel de Streamlit vía CLI; frontend React 19 en `frontend/` para flujos de trabajo de análisis y evaluación locales
- **HackBench** - conjunto de datos de referencia estandarizado (más de 4,300 trayectorias, 9 categorías)

## Stack Tecnológico

- **Núcleo (Core):** Python 3.9+, PyTorch, Hugging Face Transformers, DistilBERT, scikit-learn
- **Backend:** FastAPI, Uvicorn, Typer (CLI), Pydantic, Streamlit
- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS v4, Recharts
- **Datos:** NumPy, Pandas, SciPy, Ruptures, SQLite
- **Herramientas:** Ruff, MyPy, pytest, Rich

## Resultados

| Métrica | Prueba Held-Out | CV 5-Plegado |
|--------|:---:|:---:|
| Puntuación F1 | 89.7% | 87.4% +/- 2.9% |
| Precisión | 89.7% | 91.0% +/- 2.6% |
| Recall | 89.7% | 84.2% +/- 4.0% |
| Exactitud | 99.3% | 99.0% +/- 0.2% |

Validado en 5,391 trayectorias MALT. Consulte el [desglose completo por categoría y la metodología](paper/RewardHackWatch.pdf).

| Método | F1 |
|--------|:---:|
| **DistilBERT (Nosotros)** | **89.7%** |
| Patrones Regex | 4.9% |
| BoW + LogReg | 7.0% |
| Coincidencia de Palabras Clave | 0.1% |

## Capturas de Pantalla

<details>
<summary>Ver las 9 páginas</summary>

| | |
|:---:|:---:|
| ![Dashboard](assets/screenshots/dashboard.png) **Panel Principal** | ![Quick Analysis](assets/screenshots/quick-analysis.png) **Análisis Rápido** |
| ![Timeline](assets/screenshots/timeline.png) **Línea de Tiempo** | ![Alerts](assets/screenshots/alerts.png) **Alertas** |
| ![Cross-Model](assets/screenshots/cross-model.png) **Comparación entre Modelos** | ![CoT Viewer](assets/screenshots/cot-viewer.png) **Visor CoT** |
| ![Eval Workbench](assets/screenshots/eval-workbench.png) **Mesa de Trabajo de Evaluación** | ![Session Logs](assets/screenshots/session-logs.png) **Registros de Sesión** |
| ![Settings](assets/screenshots/settings.png) **Configuración** | |

</details>

## Configuración

```bash
export ANTHROPIC_API_KEY="your-key"        # Para el juez LLM de Anthropic
export OPENAI_API_KEY="your-key"           # Para el juez LLM de OpenAI
export OLLAMA_HOST="http://localhost:11434" # Para el juez Llama local
export RHW_HACK_THRESHOLD="0.02"           # Umbral de detección
```

```python
# Calibrar con sus propios datos (recomendado)
detector = RewardHackDetector()
detector.calibrate_threshold(clean_trajectories, percentile=99)
```

## Documentación

- [Arquitectura](ARCHITECTURE.md) - diseño del sistema
- [Artículo](paper/RewardHackWatch.pdf) - paper de investigación
- [Modelo en HuggingFace](https://huggingface.co/aerosta/rewardhackwatch) - clasificador DistilBERT preentrenado
- [Página del Proyecto](https://aerosta.github.io/rewardhackwatch) - resumen y enlaces

<details>
<summary><strong>Arquitectura</strong></summary>

```
rewardhackwatch/
  core/
    detectors/           # Detección por patrón (45 regex) + ML (DistilBERT) + AST
    analyzers/           # Análisis CoT, complejidad, detección de ofuscación
    judges/              # Jueces LLM (Anthropic, OpenAI, Ollama)
    trackers/            # Rastreo RMGI, punto de cambio PELT, RMGI Causal
    calibration.py       # Calibración dinámica de umbrales
  training/              # Pipelines de DistilBERT + AttentionClassifier
  eval/                  # Cargador JSONL, puntuación por rúbrica, análisis por lotes
  experiments/           # Estudio de transferencia, ataques de evasión
  rhw_bench/             # Conjunto de datos HackBench, generadores, casos de prueba
  api/                   # Servidor REST FastAPI
  cli.py                 # Interfaz de línea de comandos
frontend/                # Panel React 19 + TypeScript + Tailwind CSS v4
paper/                   # Artículo de investigación
```

</details>

<details>
<summary><strong>Pipeline de Detección</strong></summary>

```
Entrada de Trayectoria
       |
       v
+------------------+
|  Capa de Detección |
+------------------+
| Clasificador ML     | <-- Señal principal (89.7% F1, DistilBERT)
| Detector de Patrones| <-- 45 patrones regex (interpretabilidad)
| Analizador AST      | <-- Análisis de estructura de código
+--------+---------+
         |
         v
+------------------+
|  Capa de Análisis  |
+------------------+
| Analizador CoT      | <-- Detección de engaños en el razonamiento
| Analizador de Esfuerzo| <-- Soluciones sospechosas de bajo esfuerzo
| Verificación Complejidad| <-- Detección de ofuscación
+--------+---------+
         |
         v
+------------------+
|  Capa de Rastreo   |
+------------------+
| Métrica RMGI        | <-- Correlación hack-desalineación
| RMGI Causal         | <-- Causalidad de Granger
| Detección PELT      | <-- Puntos de cambio conductuales
+--------+---------+
         |
         v
    Alerta / Informe
```

</details>

<details>
<summary><strong>Trabajos Relacionados</strong></summary>

| Herramienta | Dominio | Enfoque | Alcance |
|------|--------|----------|-------|
| **RewardHackWatch** | Agentes LLM | Multicapa (ML + patrón + RMGI) | Detección en tiempo de ejecución + rastreo de generalización |
| RewardScope | RL Clásico | Análisis de modelo de recompensa | Depuración de función de recompensa |
| OpenAI CoT Monitor | Modelos de razonamiento | Escaneo de cadena de pensamiento | Monitoreo interno (no es código abierto) |
| SHADE-Arena | Agentes LLM | Marco de evaluación | Referencia, no detección |
| MALT | Agentes LLM | Conjunto de datos de trayectorias | Solo datos, sin detector |

</details>

<details>
<summary><strong>Figuras</strong></summary>

| Transición RMGI | Arquitectura |
|:---:|:---:|
| ![RMGI transition detection](figures/fig1_transition_plot.png) | ![System architecture](figures/fig2_architecture.png) |
| **Referencia (Benchmark)** | **Categorías** |
| ![Baseline comparison](figures/fig3_benchmark_comparison.png) | ![Per-category F1](figures/fig4_category_performance.png) |
| **Umbral** | **Calibración** |
| ![Threshold sensitivity](figures/fig5_threshold_sensitivity.png) | ![Model calibration](figures/fig6_calibration.png) |

</details>

## Licencia

Licencia Apache 2.0 - consulte [LICENSE](LICENSE) para obtener más detalles.

Derechos de autor 2025-2026 Aerosta

## Citación

```bibtex
@software{aerosta2025rewardhackwatch,
  title={RewardHackWatch: Runtime Detection of Reward Hacking and
         Misalignment Generalization in LLM Agents},
  author={Aerosta},
  year={2025},
  url={https://github.com/aerosta/rewardhackwatch}
}
```

---

Construido para investigadores e ingenieros de seguridad en IA. Los comentarios y contribuciones son bienvenidos. Siga las actualizaciones en [X @aerosta_ai](https://x.com/aerosta_ai).
