"""
RewardHackWatch FastAPI Backend.
"""

from __future__ import annotations

"""

Provides REST API and WebSocket endpoints for:
- Single trajectory analysis
- Batch trajectory analysis
- Health monitoring
- Statistics retrieval
- Live monitoring stream
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rewardhackwatch.core.analyzers import CoTAnalyzer, EffortAnalyzer, ObfuscationDetector
from rewardhackwatch.core.detectors import ASTDetector, PatternDetector
from rewardhackwatch.core.monitors import AlertSystem, AlertSystemConfig
from rewardhackwatch.core.trackers import GeneralizationTracker


# Pydantic models for API
class TrajectoryInput(BaseModel):
    """Input model for trajectory analysis."""

    steps: list[dict[str, Any]] | None = Field(default=None, description="List of trajectory steps")
    cot_traces: list[str] | None = Field(default=None, description="Chain-of-thought traces")
    code_outputs: list[str] | None = Field(default=None, description="Code outputs")
    task: str | None = Field(default=None, description="Task description")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "cot_traces": ["Let me solve this problem..."],
                "code_outputs": ["def solve(): return 42"],
                "task": "Implement a function that returns 42",
            }
        }


class AnalysisResult(BaseModel):
    """Output model for analysis results."""

    hack_score: float = Field(description="Reward hacking score (0-1)")
    generalization_risk: float = Field(description="Risk of generalization to misalignment (0-1)")
    deception_score: float = Field(description="Deceptive reasoning score (0-1)")
    effort_suspicion: float = Field(description="Low-effort shortcut suspicion (0-1)")
    risk_level: str = Field(description="Overall risk level")
    detections: list[dict[str, Any]] = Field(description="Individual detections")
    alerts: list[dict[str, Any]] = Field(description="Triggered alerts")
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchInput(BaseModel):
    """Input model for batch analysis."""

    trajectories: list[TrajectoryInput] = Field(description="List of trajectories to analyze")
    parallel: bool = Field(default=True, description="Run analyses in parallel")


class BatchResult(BaseModel):
    """Output model for batch analysis."""

    results: list[AnalysisResult] = Field(description="Analysis results for each trajectory")
    summary: dict[str, Any] = Field(description="Aggregate statistics")
    processing_time_ms: float = Field(description="Total processing time in milliseconds")


class HealthStatus(BaseModel):
    """Health check response."""

    status: str
    version: str
    uptime_seconds: float
    components: dict[str, str]


class StatsResponse(BaseModel):
    """Statistics response."""

    total_analyses: int
    total_alerts: int
    alerts_by_level: dict[str, int]
    alerts_by_source: dict[str, int]
    average_scores: dict[str, float]
    trend: dict[str, Any]


# Global state
class AppState:
    def __init__(self):
        self.start_time = datetime.now()
        self.pattern_detector = PatternDetector()
        self.ast_detector = ASTDetector()
        self.cot_analyzer = CoTAnalyzer()
        self.effort_analyzer = EffortAnalyzer()
        self.obfuscation_detector = ObfuscationDetector()
        self.tracker = GeneralizationTracker()
        self.alert_system = AlertSystem(
            AlertSystemConfig(
                db_path="rewardhackwatch_api.db",
                enable_console=False,
                enable_file_log=True,
                log_file_path="api_alerts.log",
            )
        )
        self.active_websockets: list[WebSocket] = []
        self.analysis_count = 0


state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global state
    state = AppState()
    yield
    # Cleanup
    state = None


# Create FastAPI app
app = FastAPI(
    title="RewardHackWatch API",
    description="""
    Real-time detection of reward hacking → misalignment generalization in LLM agents.

    ## Features
    - **Pattern Detection**: Regex + AST-based detection of known reward hacks
    - **CoT Analysis**: Deceptive reasoning pattern detection
    - **Effort Analysis**: TRACE-style low-effort shortcut detection
    - **Generalization Tracking**: PELT-based transition point detection
    - **Multi-Judge System**: Claude 4.5 Opus + Llama 3.1 cross-validation

    ## Endpoints
    - `POST /analyze` - Analyze a single trajectory
    - `POST /analyze/batch` - Analyze multiple trajectories
    - `GET /status` - Health check
    - `GET /stats` - Detection statistics
    - `WebSocket /ws/monitor` - Live monitoring stream
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def analyze_trajectory_sync(trajectory: dict[str, Any]) -> AnalysisResult:
    """Synchronous trajectory analysis."""
    global state
    if not state:
        raise RuntimeError("Application not initialized")

    # Run detectors
    pattern_result = state.pattern_detector.detect(trajectory)
    ast_result = state.ast_detector.detect(trajectory)

    # Combine hack scores
    hack_score = max(pattern_result.score, ast_result.score)

    # Run CoT analyzer
    cot_result = state.cot_analyzer.analyze(trajectory)
    deception_score = cot_result.deception_score

    # Run effort analyzer
    effort_result = state.effort_analyzer.analyze(trajectory)
    effort_suspicion = effort_result.effort_suspicion_score

    # Run generalization tracker
    tracker_result = state.tracker.analyze_trajectory(trajectory)
    risk_map = {"none": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
    generalization_risk = risk_map.get(tracker_result.risk_level, 0.0)

    # Determine overall risk level
    max_score = max(hack_score, generalization_risk, deception_score, effort_suspicion)
    if max_score >= 0.8:
        risk_level = "critical"
    elif max_score >= 0.6:
        risk_level = "high"
    elif max_score >= 0.4:
        risk_level = "medium"
    elif max_score >= 0.2:
        risk_level = "low"
    else:
        risk_level = "none"

    # Collect detections
    detections = []
    for d in pattern_result.detections:
        detections.append(
            {
                "source": "pattern_detector",
                "pattern": d.pattern_name,
                "description": d.description,
                "location": d.location,
                "confidence": d.confidence,
            }
        )
    for d in ast_result.detections:
        detections.append(
            {
                "source": "ast_detector",
                "pattern": d.pattern_name,
                "description": d.description,
                "location": d.location,
                "confidence": d.confidence,
            }
        )

    # Process through alert system
    alert_result = state.alert_system.process_analysis(
        hack_score=hack_score,
        generalization_risk=generalization_risk,
        deception_score=deception_score,
        cot_consistency_score=cot_result.consistency_score,
        file_path=f"api_analysis_{state.analysis_count}",
        detector_results=[pattern_result.to_dict(), ast_result.to_dict()],
        cot_analysis_result=cot_result.to_dict(),
    )

    state.analysis_count += 1

    # Format alerts
    alerts = [
        {
            "level": a.level.value,
            "source": a.source.value,
            "message": a.message,
            "value": a.value,
        }
        for a in alert_result.alerts_triggered
    ]

    return AnalysisResult(
        hack_score=hack_score,
        generalization_risk=generalization_risk,
        deception_score=deception_score,
        effort_suspicion=effort_suspicion,
        risk_level=risk_level,
        detections=detections,
        alerts=alerts,
        metadata={
            "cot_patterns": len(cot_result.suspicious_patterns),
            "effort_shortcuts": effort_result.metrics.shortcuts_detected,
            "tracker_correlation": tracker_result.correlation,
        },
    )


@app.post("/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_trajectory(trajectory: TrajectoryInput) -> AnalysisResult:
    """
    Analyze a single trajectory for reward hacking and misalignment.

    Returns comprehensive analysis including:
    - Hack score (code-level reward hacking)
    - Generalization risk (hack→misalignment correlation)
    - Deception score (CoT deceptive reasoning)
    - Effort suspicion (low-effort shortcuts)
    - Individual detections and alerts
    """
    try:
        traj_dict = trajectory.model_dump(exclude_none=True)
        result = analyze_trajectory_sync(traj_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch", response_model=BatchResult, tags=["Analysis"])
async def analyze_batch(batch: BatchInput) -> BatchResult:
    """
    Analyze multiple trajectories in batch.

    Optionally runs analyses in parallel for improved performance.
    Returns individual results and aggregate statistics.
    """
    start_time = datetime.now()

    try:
        results = []

        if batch.parallel:
            # Run in parallel using asyncio
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(None, analyze_trajectory_sync, t.model_dump(exclude_none=True))
                for t in batch.trajectories
            ]
            results = await asyncio.gather(*tasks)
        else:
            # Run sequentially
            for t in batch.trajectories:
                result = analyze_trajectory_sync(t.model_dump(exclude_none=True))
                results.append(result)

        # Calculate summary
        n = len(results)
        summary = {
            "total_trajectories": n,
            "avg_hack_score": sum(r.hack_score for r in results) / n if n > 0 else 0,
            "avg_deception_score": sum(r.deception_score for r in results) / n if n > 0 else 0,
            "max_hack_score": max(r.hack_score for r in results) if results else 0,
            "trajectories_with_alerts": sum(1 for r in results if r.alerts),
            "total_alerts": sum(len(r.alerts) for r in results),
            "risk_distribution": {
                "critical": sum(1 for r in results if r.risk_level == "critical"),
                "high": sum(1 for r in results if r.risk_level == "high"),
                "medium": sum(1 for r in results if r.risk_level == "medium"),
                "low": sum(1 for r in results if r.risk_level == "low"),
                "none": sum(1 for r in results if r.risk_level == "none"),
            },
        }

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return BatchResult(
            results=results,
            summary=summary,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=HealthStatus, tags=["Monitoring"])
async def health_check() -> HealthStatus:
    """
    Check API health status.

    Returns component status and uptime information.
    """
    global state
    if not state:
        raise HTTPException(status_code=503, detail="Service not initialized")

    uptime = (datetime.now() - state.start_time).total_seconds()

    return HealthStatus(
        status="healthy",
        version="0.1.0",
        uptime_seconds=uptime,
        components={
            "pattern_detector": "ready",
            "ast_detector": "ready",
            "cot_analyzer": "ready",
            "effort_analyzer": "ready",
            "alert_system": "ready",
            "database": "connected",
        },
    )


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_statistics() -> StatsResponse:
    """
    Get detection statistics.

    Returns aggregate statistics including alert counts,
    average scores, and trend analysis.
    """
    global state
    if not state:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = state.alert_system.get_statistics()

    return StatsResponse(
        total_analyses=stats.get("total_analyses", 0),
        total_alerts=stats.get("total_alerts", 0),
        alerts_by_level=stats.get("alerts_by_level", {}),
        alerts_by_source=stats.get("alerts_by_source", {}),
        average_scores=stats.get("average_scores", {}),
        trend=stats.get("trend", {}),
    )


@app.get("/alerts", tags=["Monitoring"])
async def get_alerts(
    limit: int = 100,
    level: str | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """
    Get recent alerts.

    Args:
        limit: Maximum number of alerts to return
        level: Filter by alert level (warning, critical)
        source: Filter by alert source (hack_score, deception_score, etc.)
    """
    global state
    if not state:
        raise HTTPException(status_code=503, detail="Service not initialized")

    from rewardhackwatch.core.monitors import AlertLevel, AlertSource

    level_filter = None
    source_filter = None

    if level:
        try:
            level_filter = AlertLevel(level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid level: {level}")

    if source:
        try:
            source_filter = AlertSource(source)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid source: {source}")

    alerts = state.alert_system.get_alerts(
        limit=limit,
        level=level_filter,
        source=source_filter,
    )

    return alerts


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


ws_manager = ConnectionManager()


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """
    WebSocket endpoint for live monitoring.

    Streams real-time analysis results and alerts to connected clients.
    Clients can also send trajectories for immediate analysis.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Receive trajectory for analysis
            data = await websocket.receive_json()

            if data.get("type") == "analyze":
                trajectory = data.get("trajectory", {})
                result = analyze_trajectory_sync(trajectory)

                # Send result back
                await websocket.send_json(
                    {
                        "type": "analysis_result",
                        "timestamp": datetime.now().isoformat(),
                        "result": result.model_dump(),
                    }
                )

                # Broadcast to all connected clients if alerts
                if result.alerts:
                    await ws_manager.broadcast(
                        {
                            "type": "alert",
                            "timestamp": datetime.now().isoformat(),
                            "alerts": result.alerts,
                        }
                    )

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
