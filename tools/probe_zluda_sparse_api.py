"""Check cuSPARSE entry points used by PyTorch beyond library initialization."""
import ctypes
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
library = ctypes.WinDLL(str(root / ".local-zluda/zluda/cusparse64_11.dll"))
handle = ctypes.c_void_p()
library.cusparseCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
library.cusparseCreate.restype = ctypes.c_int
library.cusparseSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
library.cusparseSetStream.restype = ctypes.c_int
library.cusparseSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
library.cusparseSetPointerMode.restype = ctypes.c_int
library.cusparseDestroy.argtypes = [ctypes.c_void_p]
library.cusparseDestroy.restype = ctypes.c_int
result = {"cusparseCreate": library.cusparseCreate(ctypes.byref(handle))}
if result["cusparseCreate"] == 0:
    result["cusparseSetStream"] = library.cusparseSetStream(handle, None)
    result["cusparseSetPointerMode"] = library.cusparseSetPointerMode(handle, 0)
    result["cusparseDestroy"] = library.cusparseDestroy(handle)
library.cusparseGetErrorString.argtypes = [ctypes.c_int]
library.cusparseGetErrorString.restype = ctypes.c_char_p
report = {name: {"code": code, "message": library.cusparseGetErrorString(code).decode()}
          for name, code in result.items()}
print(json.dumps(report, indent=2), flush=True)
(root / "artifacts/zluda/sparse-api.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
raise SystemExit(0 if all(code == 0 for code in result.values()) else 1)
