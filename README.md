# Chope MLE Assignment

# Set up
Set up the Python environment via conda:

1. <a href="https://docs.conda.io/en/latest/miniconda.html">Install Miniconda Python 3.8</a> and place at: `~/miniconda3/etc/profile.d/conda.sh`
2. Grant permission to scripts: `chmod -R +x ./scripts`
3. Run set up script: `./scripts/01_setup.sh`

# How to run the repo
1. Activate env: `conda activate chope-mle`

```
# Train
cd pipeline
export TAG=v0.1 \
    && export IMAGE=chope_mle/rec_train \
    && export CTN_REC_SERVICE_NAME=ctn-rec-service \
    && export VOLUME_NAME=data-rec \
    && export INPUT_RESERVATION_PATH=data/datasets/reservations.csv \
    && export OUTPUT_MODEL_DIR=data/models/lfm \
    && export LFM_LOSS=warp \
    && export LFM_NUM_COMPONENTS=10
# Build the service
# Test API
curl -X POST "http://localhost:5002/recommend" -H "accept: */*" -H "Content-Type: application/json" -d "{\"hashed_email\":\"c625da72fbbc9e72b10e6015a5c7bee8\"}"
# Clean up
make clean-up
```
