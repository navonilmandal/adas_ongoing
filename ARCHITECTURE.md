flowchart LR

    A[Video / Camera Input] --> B[Frame Preprocessing]

    B --> C{Per-frame Inference}

    subgraph Inference
        C1[YOLOv8 Detection]
        C2[YOLOv8 Segmentation]
        C3[Traffic Light Color Detector]
    end

    C --> C1
    C --> C2
    C --> C3

    C --> D[Post-processing]

    D --> E[Tracking (DeepSORT)]
    D --> F[Lane Detection + Hood Removal]

    E --> G[Speed & TTC Calculator]
    F --> G

    G --> H[Event Manager & Alarms]

    H --> I[HUD Renderer]

    I --> J[Output Video / Frames]
    H --> K[CSV / JSON Logs]
    H --> L[Realtime API (REST/MQTT)]

    subgraph Training Pipeline
        M[Raw IDD Dataset]
        N[Augmentation & Preprocessing]
        O[YOLOv8 Training (seg/det)]
        P[Model Weights (best.pt)]
    end

    M --> N
    N --> O
    O --> P
    P --> C1
    P --> C2
