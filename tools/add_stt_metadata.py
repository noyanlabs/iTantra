"""Adds the metadata sherpa-onnx expects to the IndicConformer CTC model.

Usage (once):  pip install onnx && python tools/add_stt_metadata.py ~/Downloads/stt/model.int8.onnx
Writes assets/stt/model.int8.onnx
"""
import sys
import onnx

src = sys.argv[1]
m = onnx.load(src)
for k, v in {
    "vocab_size": "257",          # 256 tokens + blank (last id)
    "subsampling_factor": "8",    # Conformer-Large (IndicConformer) uses 8x
    "normalize_type": "per_feature",
    "is_giga_am": "0",
}.items():
    e = m.metadata_props.add()
    e.key, e.value = k, v
onnx.save(m, "assets/stt/model.int8.onnx")
print("saved assets/stt/model.int8.onnx")
