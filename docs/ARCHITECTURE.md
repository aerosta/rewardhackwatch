# RewardHackWatch Architecture

## System Overview

```mermaid
graph TB
    subgraph Input["Input Layer"]
        A[Agent Trajectory]
        B[CoT Traces]
        C[Code Outputs]
    end

    subgraph Detection["Detection Layer"]
        D[Pattern Detector<br/>40+ regex patterns]
        E[ML Classifier<br/>DistilBERT, F1: 89.7%]
    end

    subgraph Tracking["Generalization Tracking"]
        F[RMGI Tracker<br/>Rolling correlation]
        G[PELT Changepoint<br/>Transition detection]
    end

    subgraph Output["Output Layer"]
        H[Risk Level]
        I[Alerts]
        J[Transition Points]
    end

    A --> D
    B --> D
    B --> E
    C --> D
    C --> E

    D --> F
    E --> F

    F --> G
    G --> H
    G --> I
    G --> J

    style D fill:#BBDEFB
    style E fill:#90CAF9
    style F fill:#64B5F6
    style G fill:#64B5F6
    style H fill:#42A5F5
    style I fill:#42A5F5
    style J fill:#42A5F5
```

## Component Details

### Layer 1: Pattern Detection

The pattern detector uses 40+ regex patterns to identify known reward hacking signatures:

```mermaid
graph LR
    subgraph Patterns["Pattern Categories"]
        P1[Exit Bypass<br/>sys.exit, os._exit]
        P2[Test Manipulation<br/>assert True, skip]
        P3[Result Faking<br/>return True, hardcoded]
        P4[Mock Exploitation<br/>monkeypatch, mock]
        P5[CoT Red Flags<br/>deceptive intent]
    end

    Input[Text Input] --> P1
    Input --> P2
    Input --> P3
    Input --> P4
    Input --> P5

    P1 --> Score[Suspicion Score]
    P2 --> Score
    P3 --> Score
    P4 --> Score
    P5 --> Score
```

### Layer 2: ML Classification

The ML classifier uses a fine-tuned DistilBERT model:

```mermaid
graph LR
    subgraph Model["DistilBERT Classifier"]
        T[Tokenizer<br/>512 max tokens] --> B[DistilBERT<br/>66M params]
        B --> H[Classification Head]
        H --> S[Sigmoid<br/>P(hack)]
    end

    Input[Trajectory Text] --> T
    S --> |threshold=0.02| Decision[Hack / Clean]
```

**Training Configuration:**
- Dataset: 5,391 MALT trajectories (3.6% positive rate)
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Optimizer: AdamW

### Layer 3: RMGI Tracking

The RMGI (Reward-Misalignment Generalization Index) tracks correlation between behaviors:

```mermaid
graph TB
    subgraph Computation["RMGI Computation"]
        H[Hack Scores] --> W[Sliding Window<br/>size=10]
        M[Misalign Scores] --> W
        W --> C[Pearson Correlation]
        C --> R[RMGI Value<br/>-1 to 1]
    end

    subgraph Detection["Transition Detection"]
        R --> |RMGI > 0.7| A[Alert]
        R --> |3 consecutive| T[Transition Point]
    end

    subgraph PELT["PELT Algorithm"]
        HS[Historical Scores] --> P[Changepoint Detection]
        P --> TP[Transition Points]
    end
```

## Data Flow

```mermaid
sequenceDiagram
    participant Agent
    participant API
    participant Pattern
    participant ML
    participant RMGI
    participant Dashboard

    Agent->>API: POST /analyze
    API->>Pattern: detect(text)
    API->>ML: predict(text)
    Pattern-->>API: patterns found
    ML-->>API: hack score
    API->>RMGI: update(hack_score, misalign_score)
    RMGI-->>API: correlation, transitions
    API-->>Agent: risk_level, alerts
    API->>Dashboard: stream updates
```

## File Structure

```
rewardhackwatch/
├── core/
│   ├── detectors/
│   │   ├── pattern_detector.py   # Regex-based detection
│   │   ├── ml_detector.py        # DistilBERT inference
│   │   └── ast_detector.py       # AST-based analysis
│   └── trackers/
│       └── generalization_tracker.py  # RMGI computation
├── training/
│   ├── train_distilbert.py       # Model training
│   └── model_loader.py           # Inference utilities
├── api/
│   └── main.py                   # FastAPI server
└── dashboard/
    └── app.py                    # Streamlit UI
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Inference latency | ~50ms (CPU) |
| Memory footprint | ~500MB |
| Throughput | ~200 traj/sec |
| F1 Score | 89.7% |
| 95% CI | [84.8%, 90.0%] |
