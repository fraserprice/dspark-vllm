ARG BASE=voipmonitor/vllm:eldritch-final-vbfaa36b-b12x284a2ea-kimi-specdcp-cu132-20260627
FROM ${BASE}

COPY overlay/vllm /opt/venv/lib/python3.12/site-packages/vllm

RUN find /opt/venv/lib/python3.12/site-packages/vllm -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && python -c "import vllm; from vllm.config.speculative import SpeculativeConfig; print('config OK')" \
 && python -c "from vllm.models.deepseek_v4 import DSparkDraftModel; print('model OK')" \
 && python -c "from vllm.v1.spec_decode.dspark import DSparkProposer; print('proposer OK')" \
 && python -c "import vllm.v1.spec_decode.llm_base_proposer; print('base proposer OK')" \
 && python -c "import vllm.v1.worker.gpu_model_runner; print('runner OK')" \
 && python -c "import ast; ast.parse(open('/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/model.py').read()); print('model.py parse OK')"
