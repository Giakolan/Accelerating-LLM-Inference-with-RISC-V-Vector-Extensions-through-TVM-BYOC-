# Phase 2B — Kiwipedia BYOC Codegen / Runtime / RISC-V Model Compilation

這一階段的目的：在 **x86 Ubuntu Host** 使用已修改的 **TVM Kiwipedia backend**，將 Whisper Encoder、Decoder Prefill、Decoder with Past 編譯為 **RISC-V Linux ELF `.so`**，再傳至 Banana Pi。

> **非常重要：** [930727fre 參考專案](https://github.com/930727fre/tvm_rvv_matmul_byoc) 實際使用 `bananapi` backend 及 TVM v0.20.0；本專案公開原始碼使用 `kiwipedia` backend。目前公開的 [lewis21456/tvm-byoc-kiwipedia](https://github.com/lewis21456/tvm-byoc-kiwipedia) 編譯腳本是 Host x86 `target="llvm"`。以下標註「**可直接執行**」的是環境檢查；標註「**需要調整編譯腳本**」的是移植參考專案做法的教學，**不能誤稱不修改程式就會輸出 RISC-V `.so`**。報告後期 Dynamic Shape Pass、RVV Kernel 的完整版本與參數仍需由最終原始碼驗證。

## 1. 兩個專案的 Backend 名稱對照

| 930727fre 參考專案 | 本專題 Kiwipedia | 工作內容 |
| --- | --- | --- |
| `bananapi_codegen.cc` | `kiwipedia_codegen.cc`（檔名見既有紀錄，實際查路徑） | Relax BYOC 外部 Codegen |
| `bananapi_runtime.cc` | `kiwipedia_runtime.cc` | 外部 Runtime / Tensor / `dlopen()` |
| `libmatmul_rvv.cpp` | `libmatmul_rvv.cpp`（實際使用版本待核對） | RVV Kernel |
| `bananapi.cmake` | Kiwipedia CMake 設定（檔案名稱以本 fork 為準） | 註冊 CMake 選項與 sources |
| `bananapi.matmul` | `kiwipedia.matmul`（依腳本實際 pattern） | Graph Partition Pattern |
| `USE_BANANAPI_*`（依原 fork 配置） | `USE_KIWIPEDIA_CODEGEN`, `USE_KIWIPEDIA_RUNTIME`（既有指令） | 編譯設定 |

**不要**從參考倉庫直接把 `bananapi_runtime.cc` 覆蓋掉 `kiwipedia_runtime.cc`。參考專案的文件列出 `config.cmake`、`CMakeLists.txt`、`LibInfo.cmake`、`libinfo.cc`、`bananapi.cmake`、Codegen、Runtime 等檔案，目的是示範 TVM **CMake / Registry / BYOC** 的整合點；兩個 fork 的原始碼、註冊名稱與 ABI 不一定相容。

## 2. 尋找自訂檔案與確認註冊（可直接執行）

```bash
cd "$HOME/tvm-byoc-kiwipedia"
export TVM_HOME="$PWD/tvm"

find "$TVM_HOME" -type f \( \
  -name '*kiwipedia*' -o -name 'libmatmul*.cpp' \) | sort

grep -RIn 'KIWIPEDIA\|kiwipedia' \
  "$TVM_HOME/CMakeLists.txt" \
  "$TVM_HOME/cmake" \
  "$TVM_HOME/src/relax/backend/contrib" \
  "$TVM_HOME/src/runtime/contrib" 2>/dev/null | head -70

grep -RIn 'RemoveKiwipediaShapeParams' "$TVM_HOME/src" || true
```

應確認：

- CMake 會編進 Kiwipedia Codegen 與 Runtime 的 C++ 原始檔。
- 編譯腳本 `FuseOpsByPattern` 的 backend 前綴、`RunCodegen()`、Runtime 註冊名稱與 C++ 實作一致。
- 若部署 **Dynamic Shape Decoder with Past**，報告中 `RemoveKiwipediaShapeParams` 在 `RunCodegen()` **之前**執行，Runtime 從實際 `DLTensor` 取得尺寸並驗證輸出 buffer。

## 3. 重新建置 Host TVM（有修改 C++ 時必要）

```bash
cd "$HOME/tvm-byoc-kiwipedia"
source .venv/bin/activate

cmake -S tvm -B tvm/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_LLVM=ON \
  -DUSE_KIWIPEDIA_CODEGEN=ON \
  -DUSE_KIWIPEDIA_RUNTIME=ON
cmake --build tvm/build --parallel 4

source scripts/env.local.sh  # 若未合併文件包，改 source env_tvm.sh
python - <<'PY'
import tvm
print(tvm.__file__)
print('LLVM', tvm.get_global_func('target.build.llvm', True))
print('Kiwipedia Runtime', tvm.get_global_func('runtime.kiwipedia_runtime_create', True))
PY
```

此時 `libtvm.so` 仍是 **x86 Host** 的 TVM 編譯器／Runtime Library。**不要將它放到 Banana Pi**。板端需編譯 RISC-V native 版本或使用與板端相容的 RISC-V runtime。

## 4. 參考專案的 RISC-V 匯出機制（需要調整本 fork 的 Python 程式）

原參考版 `compile_encoder.py`／`compile_decoder.py`／`compile_decoder_with_past.py` 的關鍵是：

1. `tvm.relax.frontend.onnx.from_onnx(...)` 建立 Relax 模組。
2. 依指定 backend pattern 做 `FuseOpsByPattern`、`RunCodegen()`。
3. `relax.build(..., target="llvm -mtriple=riscv64-unknown-linux-gnu ...")`，使 LLVM 產生 RISC-V 目標程式。
4. `ex.export_library(..., fcompile=riscv_fcompile)`，使用 `riscv64-unknown-linux-gnu-g++` **交叉連結** `.so`。

以下是**用來移植的程式片段，不是要求直接覆寫整份腳本**：

```python
# 在需要交叉編譯的 compile_*.py 中，依實際程式結構加入
import os
from tvm.contrib import cc

RISCV_TARGET = (
    'llvm -mtriple=riscv64-unknown-linux-gnu '
    '-mattr=+m,+a,+f,+d,+c'
)


def riscv_fcompile(file_name, files, options=None, **kwargs):
    extra = list(options or [])
    extra.extend(['-march=rv64gcv', '-mabi=lp64d'])
    return cc.create_shared(
        file_name, files,
        options=extra,
        cc=os.environ.get(
            'RISCV_CXX', 'riscv64-unknown-linux-gnu-g++'
        ),
        **kwargs,
    )

# 下列 ex 必須是原腳本用正確的 Relax IR、Pass 與 RunCodegen 建好的結果
# ex = relax.build(mod, RISCV_TARGET)
# ex.export_library(output_path, fcompile=riscv_fcompile)
```

> **注意細節：** 參考版的 LLVM target `-mattr` 為 `+m,+a,+f,+d,+c`，沒有直接要求 TVM 為每段程式產生 RVV；**外部 RVV MatMul Kernel 的向量化來自用 RVV 旗標編譯的 `libmatmul_rvv.cpp`**。此處刻意保留該分工，不把「模型 ELF 是 RISC-V」誤當成「每個運算都用 RVV」。目標 ISA、sysroot、glibc 要與板端相容。

如果某版 LLVM 需要不同的 target 字串、或 `cc.create_shared` 接受的參數 API 改變，應依 **該 fork 的 TVM 版本**核對調整；此片段僅展示參考程式的結構與必要切換點。

## 5. Decoder Dynamic Shape 是獨立問題（不可省略）

參考版 `compile_decoder_with_past.py` 具有 RISC-V target 與 `fcompile`，但是其中 `FuseOpsByPattern` 及 `RunCodegen()` 為**註解**。它**不能作為本專題「Decoder with Past 已完成 Kiwipedia BYOC + RVV」的證據**。

本專題報告所述的完整版本還需要：

- `RemoveKiwipediaShapeParams` Pass 修正額外 `tir_vars: R.Shape(...)` 參數及呼叫端。
- 在 Runtime 執行時由 `DLTensor` 讀取實際 M/K/N，不使用編譯階段不確定的 `-1` 維度。
- 透過 `GetInputs()`、`EntryID` 正確對應輸入 A/B、輸出 C，並檢查輸出元素數量。
- 加入逐 Token 變動的 KV cache 長度的正確性測試。

取得最終原始碼後，應先核對這些邏輯，再把 three `compile_*.py` 轉成跨平台版本；**不能只改 `target` 一行就聲稱 Dynamic Shape BYOC 正常**。

## 6. 確認三個腳本是否已改為 RISC-V（可直接執行）

```bash
cd "$HOME/tvm-byoc-kiwipedia/whisper-tiny/onnx"

grep -nE 'mtriple|target=|riscv_fcompile|fcompile|RunCodegen|FuseOpsByPattern' \
  compile_encoder.py compile_decoder.py compile_decoder_with_past.py

command -v riscv64-unknown-linux-gnu-g++
```

- 若仍看到 `target="llvm"`、未使用 `fcompile=...`：**只會產生 Host 編譯產物**，請勿傳到板端。
- 若已完整適配 RISC-V target、`fcompile`、BYOC Dynamic Shape，再依序執行：

```bash
# 僅於已完成上方「編譯腳本適配」後執行
python compile_encoder.py
python compile_decoder.py
python compile_decoder_with_past.py

file encoder_model.so decoder_model.so decoder_with_past_model.so
readelf -h encoder_model.so | grep 'Machine:'
readelf -h decoder_model.so | grep 'Machine:'
readelf -h decoder_with_past_model.so | grep 'Machine:'
```

預期全部顯示 `RISC-V`，而非 `x86-64`。**若編譯失敗，不能沿用先前的 `.so` 當成這次產物**；可以先將舊檔移到備份目錄，或比對產物時間及 checksum。

## 7. RVV Kernel 的獨立編譯（請勿把 Host x86 `.so` 誤用）

在 **Banana Pi** 本機編譯通常最直觀，請見 [板端部署第 4 步](banana_pi_deployment.md#4-編譯板端-rvv-matmul-kernel)。若希望 Host 先編成 RISC-V `.so`，可在有對應標頭、sysroot 的前提下交叉編譯（僅為**介面示例**）：

```bash
cd "$HOME/tvm-byoc-kiwipedia/tvm/src/runtime/contrib/kiwipedia"
DLPACK_DIR="$HOME/tvm-byoc-kiwipedia/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include"
test -f "$DLPACK_DIR/dlpack/dlpack.h" || exit 1

"$RISCV_CXX" -std=c++17 -O3 -shared -fPIC \
  -march=rv64gcv -mabi=lp64d \
  -I"$DLPACK_DIR" \
  libmatmul_rvv.cpp -o libmatmul_rvv_riscv.so

file libmatmul_rvv_riscv.so
```

此命令**只在最終 Kernel 確實位於 `libmatmul_rvv.cpp`、能獨立連結且符號與 Runtime 相容時成立**。外部 `dlopen()` 目標檔名要在板端透過 `KIWIPEDIA_MATMUL_SO`（或最終 Runtime 版本實際使用的環境變數）設為絕對路徑。若 Kernel 使用額外的 common headers、OpenMP、專案檔案，需同步加入。

## 8. 驗證 RVV 指令是否存在

```bash
"${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-objdump" \
  -d libmatmul_rvv_riscv.so | \
  grep -E 'vsetvli|vsetivli|vle32|vse32|vfmacc' | head -30
```

`objdump` 只能確認輸出 ELF 中含相關指令，**不等於模型 Runtime 一定有調用該 Kernel**；部署後仍須正確性驗證與 Profiling。

## 9. 使用參考專案的訣竅

[930727fre 的 `set_up_bananapi_rutime_and_codegen.md`](https://github.com/930727fre/tvm_rvv_matmul_byoc/blob/main/set_up_bananapi_rutime_and_codegen.md) 清楚說明選用 **`dlopen()`**，原因是讓 x86 編譯 TVM 時不必靜態編入 RISC-V Kernel，而改在板端用本機 `g++` 編譯 RVV MatMul。這個設計思想可以參考，但實際函式名稱、`dlopen()` 位置、`libmatmul` 符號必須依本專題的 `kiwipedia_runtime.cc` 為準。

[← README](../README.md) · [下一步：Banana Pi 部署 →](banana_pi_deployment.md)
