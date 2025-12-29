# property_prediction_end_to_end

This repository contains an ongoing project used to apply specific libraries in a practical codebase. The project focuses on end-to-end development, including backend logic, API design and GUI integration.

The goal is to implement an end-to-end solution prototype for the prediction of molecular properties.

In the current version, models and associated metrics are just placeholders. Model performance has not been optimized or evaluated and predictions are not reliable.

## Features (implemented)
- Prediction pipeline for inference (using placeholder models).
- API connecting the frontend and prediction pipeline.
- Interactive GUI (inference) for data upload and visualizing prediction results.

## Planned features
- Workflow for model selection
- Workflow for model training and evaluation
- Interactive GUI(s) for model selection and training
- API connecting the model selection and training GUI and the corresponding pipelines


## Installation & usage
Prerequisites:
- Git
- Docker (and Docker Compose)

Steps:
1. Clone the repository and change into it:
   git clone https://github.com/your-org/property_prediction_end_to_end.git
   cd property_prediction_end_to_end

2. From the repository root start the services with Docker Compose:
   - foreground (attach logs, blocks terminal):
     docker compose up
   - detached (runs in background):
     docker compose up -d

3. Frontend URL:
   - Once services are running, open the frontend in your browser at http://localhost:3000 (port may vary depending on compose configuration).

4. To stop and remove containers:
   docker compose down