ncu --target-processes all \
    -k "regex:.*compute_attn_ws.*" \
    -s 5 -c 1 \
    --section ComputeWorkloadAnalysis \
    $(which python) example.py