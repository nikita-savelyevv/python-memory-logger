from pathlib import Path
import tempfile
import openvino as ov
import sys
import nncf
from memory_logger import MemoryLogger

# sys.argv[1] -- model dir
# sys.argv[2] -- memory log type
# sys.argv[3] -- 0/1: disable/enable mmap

model_path = Path(sys.argv[1]) / "openvino_model.xml"

memory_logger = MemoryLogger(
    Path("./logs"),
    plot_title="Weight Compression",
    interval=1e-1,
    memory_type=sys.argv[2],
    start_memory_from_zero=bool(0)
).start_logging()

core = ov.Core()
if sys.argv[3] == "0":
    core.set_property({"ENABLE_MMAP": "NO"})
model = core.read_model(model_path)
compressed_model = nncf.compress_weights(model)

with tempfile.TemporaryDirectory() as temp_dir:
    ov.save_model(compressed_model, f"{temp_dir} / model.xml")
