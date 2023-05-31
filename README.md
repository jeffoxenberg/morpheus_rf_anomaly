# Morpheus RF Anomaly Detection Pipeline

Troubleshooting the difference in inference results seen when using morpheus pipeline vs calling triton or local model directly

## Install
In base directory: `docker compose up`.  Note that the compose file assumes a gpu is present and working in docker

## Run
**Jupyter**: http://127.0.0.1:8888/ (may need to get the auth token from the compose output).  Code is in the ipynb, I changed unneeded cells to markdown so it can be run from top to bottom

**Morpheus**: `docker exec -it morpheus bash` then `cd /morpheus/rf_anomaly`.  CLI command is in cli.txt, note that we changed it from kafka in/out to CSV for testing.  Python implementation is in run.py
**Kafka**: If using kafka, need to create input and output topics: 
```
docker exec -it kafka bash
bin/kafka-topics.sh --create --topic in --bootstrap-server localhost:9092
bin/kafka-topics.sh --create --topic out --bootstrap-server localhost:9092
```
**Triton**: Model should load automatically, located in `rf_anomaly/1/xgboost.json`

## Output
- Jupyter notebook sends the same records for inference to the local XGBoost model, directly to Triton via client HTTP API, and to Kafka for Morpheus inference
- I am observing a difference in inference output between Morpheus and both local and Triton direct model outputs.  Local and Triton outputs match, Morpheus output differs
- Sometimes, I'm also observing a difference in output just in Morpheus - for example, sending same data points to Morpheus sometimes results in different inference result
- Verified row ordering is OK, maybe there is an issue with column ordering somewhere within Morpheus?