# Build from the repository root (ProteinMPNN and its weights are vendored there):
#   docker build -f tools/ecls_scorer/Dockerfile -t ecls-scorer .
# The image bundles the ProteinMPNN script and vanilla model weights; respect the
# licenses in ProteinMPNN/LICENSE when redistributing.
FROM python:3.11-slim

WORKDIR /opt
COPY tools/ecls_scorer /opt/ecls_scorer
COPY ProteinMPNN /opt/ProteinMPNN

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir /opt/ecls_scorer

ENV ECLS_MPNN_SCRIPT=/opt/ProteinMPNN/protein_mpnn_run.py \
    ECLS_MPNN_WEIGHTS=/opt/ProteinMPNN/vanilla_model_weights

ENTRYPOINT ["ecls-score"]
