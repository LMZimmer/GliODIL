docker run --rm --gpus '"device=5"' \
           --entrypoint python3 \
           gliodilgpu:latest - <<'PY'
import tensorflow as tf, os
print("TensorFlow", tf.__version__)
print("Visible GPUs ->", tf.config.list_physical_devices("GPU"))
PY

