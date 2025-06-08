docker run --rm --gpus '"device=2"' \
           --entrypoint python3 \
           gliodil:latest - <<'PY'
import tensorflow as tf, os
print("TensorFlow", tf.__version__)
print("Visible GPUs ->", tf.config.list_physical_devices("GPU"))
PY

