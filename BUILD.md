# Environment Setup, Compilation & Deployment

本文件整理專題報告「研究方法及步驟」與既有開發紀錄，說明 **x86 Ubuntu 編譯主機 → TVM BYOC → RISC-V 交叉編譯 → QEMU → Banana Pi BPI-F3** 的操作順序。

> **新增：從零開始的可逐步操作文件**：
> [Phase 1：Ubuntu / TVM](compile_tvm.md) → [Phase 2A：RISC-V Toolchain](compile_riscv_toolchain.md) → [Phase 2B：Kiwipedia Codegen / RISC-V Models](set_up_bananapi_runtime_and_codegen.md) → [Phase 3：Banana Pi](banana_pi_deployment.md)。另外請見 [學長 bananapi 參考版與 Kiwipedia 差異](REFERENCE_COMPARISON.md)。本檔保留較完整的背景、QEMU、Profiling 與正確性檢查原理。


> **第一次安裝、想先在 x86 Ubuntu 編譯並執行 Whisper？請先依 [README Quick Start](../README.md#3-quick-startubuntu-從零建置到-whisper-推論) 逐步操作。** 此頁針對後續 RISC-V/RVV 實機部署。已核對的公開原始碼目前三支 `compile_*.py` 使用 `target="llvm"`，不能直接當作 RISC-V 交叉編譯指令。

> **資料可信度標示**：標記「**報告確認**」的內容來自研究報告；「**既有操作紀錄**」來自先前測試中確實出現的命令／檔名；「**範例／待核對**」代表參考設定，必須配合實際原始碼調整。公開版三支 x86 編譯腳本的**路徑與直接啟動方式已核對**，但報告最終 RVV 專用交叉編譯指令、產物與版本鎖定檔仍待補。

## 1. Architecture & Build Boundaries

```text
x86 Ubuntu Host
  ├─ Python / ONNX / Whisper / Pythia
  ├─ 自訂 TVM (Relax Pass + BYOC Codegen)
  ├─ x86 Host TVM Python Package / Library
  └─ RISC-V GNU cross compiler ──> RISC-V .so / test binary
                                      │
                           ┌──────────┴───────────┐
                           ▼                      ▼
                       QEMU user-mode        Banana Pi BPI-F3
                       correctness test      real performance test
```

- **報告確認**：Whisper-tiny 分成 Encoder、Decoder Prefill、Decoder with Past 三部分交叉編譯，並於 QEMU 驗證後部署 Banana Pi BPI-F3。
- **報告確認**：將 MatMul、Conv1D 等 BYOC 運算導向 Kiwipedia Runtime／自訂 RVV Kernel。
- **重要區分**：x86 主機上編譯的 `libtvm.so` 與 RISC-V 模型 `.so` **不是同一種架構**。需要區分「主機端編譯工具」與「板端 Runtime／模型模組」。

## 2. Ubuntu Host Preparation

### 2.1 前置條件

- x86_64 Ubuntu、可用的 C/C++ 編譯環境。
- **已修改的 TVM fork**（必須包含 Kiwipedia BYOC 整合、Dynamic Shape Pass、Runtime 修改）。
- RISC-V GNU C++ 交叉編譯器、對應 sysroot 與必要的 RVV intrinsics header。
- Python 模型匯出／編譯腳本及相對應 ONNX 模型。
- QEMU RISC-V user-mode（選用於交叉編譯後的正確性測試）。

**範例／待核對：** Ubuntu 可先安裝常見 TVM 建置工具；實際版本以 fork 為準。

```bash
sudo apt update
sudo apt install -y \
  git build-essential cmake ninja-build pkg-config \
  python3 python3-pip python3-venv \
  llvm-dev clang libtinfo-dev zlib1g-dev
```

### 2.2 Python 虛擬環境

```bash
cd <PROJECT_ROOT>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

> 模型匯出可能需要 `torch`、`transformers`、`onnx`、`numpy` 等套件，但**報告未提供精確版本**。建議依實際可用環境產生／提交 `requirements.txt` 或 lockfile；不要將上述名稱誤認為本專題已驗證的固定依賴清單。

## 3. Environment Variables（環境變數）

### 3.1 套用建議範本

```bash
cd <PROJECT_ROOT>
cp scripts/env.example.sh scripts/env.local.sh
${EDITOR:-nano} scripts/env.local.sh
source scripts/env.local.sh
bash scripts/check_environment.sh
```

`env.local.sh` 放自己的路徑與主機設定，`.gitignore` 已排除，避免將個人工作環境寫死在公開 repo 中。

| 變數 | 用途 | 範例 |
| --- | --- | --- |
| `PROJECT_ROOT` | 此專案根目錄 | `$HOME/tvm-byoc-kiwipedia` |
| `TVM_HOME` | 修改後 TVM 原始碼目錄 | `$PROJECT_ROOT/tvm` |
| `PYTHONPATH` | 讓 Python 載入此份 TVM | `$TVM_HOME/python` |
| `LD_LIBRARY_PATH` | 讓 host Python 找到 host TVM `.so` | `$TVM_HOME/build` |
| `RISCV_TOOLCHAIN_ROOT` | RISC-V GNU 工具鏈安裝前綴 | `$HOME/opt/riscv` |
| `RISCV_CXX` | RISC-V C++ 交叉編譯器 | `$RISCV_TOOLCHAIN_ROOT/bin/riscv64-unknown-linux-gnu-g++` |
| `RISCV_SYSROOT` | QEMU／交叉編譯所用 RISC-V sysroot | `$RISCV_TOOLCHAIN_ROOT/sysroot` |
| `QEMU_RISCV64` | RISC-V user-mode QEMU | `$HOME/qemu-rvv/bin/qemu-riscv64` |
| `KIWIPEDIA_MATMUL_SO` | 由 Kiwipedia Runtime 載入的 MatMul `.so` | `$TVM_HOME/src/runtime/contrib/kiwipedia/libmatmul.so` |
| `MODEL_DIR` | 如自訂 RVV 腳本需要，可另設模型目錄 | `$PROJECT_ROOT/whisper-tiny/onnx` |

**既有操作紀錄**曾使用 `riscv64-unknown-linux-gnu-g++`、`qemu-riscv64`、`-march=rv64gcv`、`-mabi=lp64d` 與 `-L <sysroot>`。上述 `$HOME/...` 是將當時個人目錄改寫為可公開重用的形式，**不是宣稱每一台主機安裝在這些路徑**。

**重要：** `LD_LIBRARY_PATH` 中的 `$TVM_HOME/build` 只在 **x86 Host** 使用。將 host `libtvm.so` 路徑帶到 Banana Pi 不會解決板端依賴；板端需要其自身 RISC-V Runtime 與共享函式庫路徑。

### 3.2 基本檢查

```bash
command -v riscv64-unknown-linux-gnu-g++
"$RISCV_CXX" --version
"$RISCV_CXX" -print-sysroot
"$QEMU_RISCV64" --version
python -c "import tvm; print(tvm.__file__)"
```

若 `-print-sysroot` 回傳其他位置，請優先核對交叉工具鏈的實際 sysroot，並同步修改 `RISCV_SYSROOT`。

### 3.3 RVV／多執行緒相關變數

- `-march=rv64gcv -mabi=lp64d` 是**編譯旗標**，不是 shell 環境變數；需由 CMake、`export_library()` 使用的 `cc` 命令或手動交叉編譯命令傳入。
- 多核心工作分配確實是研究內容，但**報告未指明 Runtime 實際使用哪一個控制執行緒數量的變數名稱**。請檢視程式：

```bash
grep -RIn --include='*.cc' --include='*.cpp' --include='*.h' \
  -E 'getenv\(|OMP_NUM_THREADS|TVM_NUM_THREADS' \
  "$TVM_HOME/src/runtime/contrib/kiwipedia" || true
```

只有確認原始碼會讀取該變數後，再將它寫入 `env.local.sh`，不要直接假設 `OMP_NUM_THREADS` 一定能控制所有 Kernel。

## 4. Build the Modified TVM Host

### 4.1 確認 BYOC 自訂檔案存在

**既有操作紀錄**已知的 Runtime／Kernel 相對路徑：

```text
tvm/src/runtime/contrib/kiwipedia/libmatmul.cpp
tvm/src/runtime/contrib/kiwipedia/libmatmul_rvv.cpp
tvm/src/runtime/contrib/kiwipedia/kiwipedia_runtime.cc
```

`kiwipedia_codegen.cc` 與 `RemoveKiwipediaShapeParams` Pass 的**完整位置尚未從這次提供的報告與原始碼檢驗**。請自行搜尋：

```bash
find "$TVM_HOME" -type f \( \
  -name 'kiwipedia_codegen.cc' -o \
  -name 'kiwipedia_runtime.cc' -o \
  -name 'libmatmul_rvv.cpp' \
\)
grep -RIn 'RemoveKiwipediaShapeParams' "$TVM_HOME/src" || true
```

`RemoveKiwipediaShapeParams` 是研究報告明確提到的自訂 Pass；它應於 `RunCodegen()` **之前**處理指定 BYOC 函式多出的 Shape 參數。

### 4.2 建置 Host TVM（範例／待核對）

```bash
source scripts/env.local.sh

# 若 fork 要求特定 build/config.cmake 選項，請先依 fork 修改
mkdir -p "$TVM_HOME/build"
if [ ! -f "$TVM_HOME/build/config.cmake" ]; then
  cp "$TVM_HOME/cmake/config.cmake" "$TVM_HOME/build/config.cmake"
fi

cmake -S "$TVM_HOME" -B "$TVM_HOME/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_LLVM=ON \
  -DUSE_KIWIPEDIA_CODEGEN=ON \
  -DUSE_KIWIPEDIA_RUNTIME=ON
cmake --build "$TVM_HOME/build" --parallel "$(nproc)"

python -c "import tvm; print('Loaded TVM from:', tvm.__file__)"
```

若自訂 BYOC 模組需要額外的 CMake source list、Runtime 開關或庫連結，須使用**既有 fork 的配置**；這裡的通用 CMake 命令無法代替缺少的 Kiwipedia 原始碼或註冊步驟。

## 5. ONNX → BYOC 模型交叉編譯

### 5.1 準備輸入

**報告確認：** Whisper-tiny 需編譯三個模組。

1. **Encoder**：音訊 Mel 特徵 → Encoder 特徵；專題有 15 秒與 30 秒 Shape Bucket。
2. **Decoder Prefill**：處理起始 Token，建立本次音訊的 Decoder 初始狀態。
3. **Decoder with Past**：接收、更新 KV cache 以逐 Token 產生輸出；涉及 Dynamic Shape。

**既有操作紀錄**的 Python 檔名：

```text
compile_encoder.py
compile_decoder.py
compile_decoder_with_past.py
inference.py
```

> 目前公開版已確認這些腳本的路徑：`whisper-tiny/onnx/compile_encoder.py`、`compile_decoder.py`、`compile_decoder_with_past.py` 及 `whisper-tiny/inference.py`；三支編譯腳本無 CLI 參數，輸出同名 `.so`。**但這些腳本在現有公開版使用 x86 `target="llvm"`，不是此文件要說明的 RISC-V/RVV 最終版本。**

```bash
find "$PROJECT_ROOT" -type f \( \
  -name 'compile_encoder.py' -o \
  -name 'compile_decoder.py' -o \
  -name 'compile_decoder_with_past.py' -o \
  -name 'inference.py' \
\)
```

### 5.2 編譯時的核心動作

```text
ONNX Model
   ↓ Import
Relax IR
   ↓ IR Pass / BYOC Graph Partition
RunCodegen（之前先執行 RemoveKiwipediaShapeParams）
   ↓ TVM + Kiwipedia External Codegen
RISC-V Target Code + RVV Runtime / Kernel
   ↓ Cross compiler / export_library
RISC-V .so
```

- ONNX 的模型版本、輸入 tensor names／shape、dynamic dimensions，需與實際 Python 程式相符。
- Host Python 執行 `compile_*.py`，由 TVM 建置 RISC-V 目標模組；**並非在 x86 上直接執行 RISC-V `.so`**。
- `decoder_with_past` 要保留 KV cache 動態長度所需的形狀資訊；報告的 Pass 只移除 BYOC 函式中**多出的 Shape 參數**，Kernel 執行時仍需從 `DLTensor` 取得實際維度。
- Kiwipedia Runtime 需使用 `GetInputs()`、`EntryID` 正確取得 A、B、C，並檢查輸出 Tensor 元素數量。

### 5.3 目前公開版 x86 編譯指令（已確認）

```bash
cd "$PROJECT_ROOT/whisper-tiny/onnx"
python compile_encoder.py
python compile_decoder.py
python compile_decoder_with_past.py
```

三個輸出分別為 `encoder_model.so`、`decoder_model.so`、`decoder_with_past_model.so`。**這三行只適用於原公開版的 x86 Host Quick Start**；欲編譯 RISC-V 必須取得相容的完整 RVV 版本、設定目標 triple、RVV 交叉編譯器與 sysroot，並確認 `export_library()` 如何連結外部 Kernel。15 秒／30 秒 Encoder 也需要與對應 ONNX Shape 一致的獨立編譯工件。

### 5.4 RVV 交叉編譯器旗標（既有操作紀錄）

先前獨立 OP Benchmark 曾實際使用下列命令格式：

```bash
"$RISCV_CXX" \
  -O3 -march=rv64gcv -mabi=lp64d \
  -DWHISPER_REQUIRE_RVV \
  benchmark_gelu_qemu.cpp gelu_rvv.cpp \
  -o benchmark_gelu_rvv
```

這是**獨立 GELU 測試程式範例**，不是三個 Whisper 模組的完整編譯指令。`benchmark_gelu_qemu.cpp` 和 `gelu_rvv.cpp` 必須存在於執行目錄；只需要在測試該 OP 時使用。

若自行編譯 RVV `.so`，通常會包含 `-fPIC -shared`、目標 ISA／ABI、TVM／DLPack Header 與對應連結依賴；**實際外部 Kernel 是否獨立 `.so`、與 TVM 模組如何連結，必須由 Codegen 與 Runtime 程式決定**，不能僅靠通用 `g++ -shared` 範例判定部署正確。

## 6. Inspect Cross-compiled Outputs

```bash
file <RISCV_MODEL_MODULE.so>
readelf -h <RISCV_MODEL_MODULE.so> | grep -E 'Class:|Machine:'
"${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-objdump" \
  -d <RISCV_TEST_BINARY> | \
  grep -E 'vsetvli|vsetivli|vle32|vse32|vfmacc' | head -30
```

- `file`／`readelf` 應顯示目標為 RISC-V ELF，而不是 x86-64。
- RVV 組語檢查應針對**真正編入 RVV Kernel 的執行檔／共享庫**；若未看到 RVV 指令，還須檢查是否確實走到目標函式或編譯選項是否生效。

## 7. QEMU Functional Validation

**既有操作紀錄**使用的 QEMU user-mode 格式：

```bash
"$QEMU_RISCV64" \
  -L "$RISCV_SYSROOT" \
  -cpu max \
  ./benchmark_gelu_rvv
```

在部分 QEMU 版本，也曾使用明確的 RVV CPU 參數（例：`-cpu rv64,v=true,vlen=128`）；**可用參數取決於 QEMU 版本**，請使用：

```bash
"$QEMU_RISCV64" -cpu help
```

正確性測試建議比較 **Scalar vs RVV** 的輸出，使用適當的浮點數容差，並測試不整除向量寬度的尾端長度。**QEMU 的執行時間不能直接代表 Banana Pi 效能**，這點在原始報告中特別說明。

## 8. Banana Pi BPI-F3 Deployment

### 8.1 傳輸檔案（範例／待核對）

```bash
# 將下列 placeholder 換成實際板端帳號、IP、編譯工件目錄
scp <RISCV_ENCODER_SO> <BOARD_USER>@<BOARD_IP>:~/whisper-tiny/
scp <RISCV_DECODER_SO> <BOARD_USER>@<BOARD_IP>:~/whisper-tiny/
scp <RISCV_DECODER_WITH_PAST_SO> <BOARD_USER>@<BOARD_IP>:~/whisper-tiny/
```

依 `inference.py` 是否需要 Processor、Tokenizer 或其他檔案，另外部署對應模型資產。若板端要用到 `libtvm_runtime.so` 或 `libtvm.so`，請提供**RISC-V 建置版本**及其相關依賴，不要複製 x86 host 版本。

### 8.2 板端檢查與推論

```bash
ssh <BOARD_USER>@<BOARD_IP>
uname -m
file ~/whisper-tiny/<RISCV_MODEL_MODULE.so>

# 視 Runtime 實際安裝位置設定，不要沿用 host 的 $TVM_HOME/build
export LD_LIBRARY_PATH="$HOME/whisper-tiny/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 於板端實際 inference.py 所在目錄執行，參數需依其 CLI 說明設定
python inference.py
```

> Banana Pi 使用的 Linux 發行版、板端 Python／TVM Runtime 安裝方式、`inference.py` CLI 並未在這份報告完整列出，**上述最後一行只是啟動位置示例**，不是可保證重現的命令。

### 8.3 測試順序

1. Encoder 輸出正確性與 Shape（30 秒 Encoder：長度 1500；15 秒 Encoder：長度 750）。
2. Decoder Prefill 是否產生可供後續生成的 KV cache。
3. Decoder with Past 在 KV 長度變化時，是否能持續生成至 EOS。
4. 確認 BYOC Kernel 確實被調用，並依 MatMul Shape 選擇 Generic／Shape-tuned 路徑。
5. 再測量 Encoder、Prefill、Generation 與 end-to-end 秒數；比較前保持輸入、暖機及計時範圍一致。

## 9. Profiling & Benchmarks

- 原報告以 Whisper `jfk.flac`（11 秒）呈現 O0–O4 的逐步改善。
- 主要 MatMul Shape：30 秒 Encoder `M=1500`、15 秒 Encoder `M=750`、Decoder 逐 token `M=1`。
- Profiling 優先檢查 `fused_relax_matmul...` 等高耗時算子，並區分單一 OP 加速與整體端到端加速。
- GELU、SiLU 等非矩陣類運算曾進行探索，但若運算占比低且調用成本高，RVV 化不一定提升整體推論速度。
- 長音訊：每個 Chunk 可重用模型與 VM，**不能直接沿用上一段音訊的 Self-Attention KV cache**。
- Pythia Benchmark 僅代表 16 tokens 的**固定長度完整 Forward Pass**，不可誤寫成 tokens/s 的自回歸生成速度。

## 10. Frequent Issues

| 現象 | 可能原因與檢查方向 |
| --- | --- |
| `tir_vars: R.Shape(...)` 導致 Codegen 型別錯誤 | 確認自訂 `RemoveKiwipediaShapeParams` Pass 是否正確在 `RunCodegen()` 前執行，且定義／呼叫參數同步修改 |
| Decoder 執行時出現 -1 或錯誤維度 | 確認 Runtime 不依賴編譯期 `shapeA / shapeB`，而從實際 `DLTensor` 取得 shape |
| MatMul 結果錯誤／buffer 寫入不符 | 檢查 `GetInputs()`、`EntryID` 與輸出元素數量驗證 |
| `ImportError` 或連結找不到 TVM | 檢查 Host `PYTHONPATH`、`LD_LIBRARY_PATH`，不要把 Host `.so` 放到板端 |
| QEMU 找不到動態載入器 | 檢查 `-L "$RISCV_SYSROOT"`、ABI、sysroot 結構是否一致 |
| 編譯出來卻沒有 RVV 指令 | 檢查 `-march`、目標 GCC、實際執行的 Kernel 及 `objdump` |
| `429: Too Many Requests` 出現在 `.cc` 第一行 | 先前曾碰到下載錯誤訊息被覆寫進 C++ 原始碼；應檢查檔案內容並從可信原始碼還原，非 C++ 語法本身問題 |
| RVV 版本在 QEMU 很快／很慢，實機相反 | QEMU 主要驗證功能；快慢需用相同輸入於 Banana Pi 實測 |

## 11. Reproduction Checklist（公開前）

- [ ] 提交修改後 TVM fork 或可重現的 patches，註明 commit
- [ ] 公開 `compile_encoder.py`、`compile_decoder.py`、`compile_decoder_with_past.py`、`inference.py`（或說明取得方式）
- [ ] 紀錄完整 CLI 與成功生成的 `.so` 實際名稱
- [ ] 紀錄 Python 套件／TVM／LLVM／GCC／QEMU 版本
- [ ] 補充 ONNX 匯出方式、模型權重來源與校驗
- [ ] 確認 Banana Pi 系統與板端 Runtime 建置方式
- [ ] 提供 Benchmark 輸入檔案、測試指令、正確性門檻及結果
- [ ] 釐清所有第三方程式碼及模型檔的授權

[← 返回 README](../README.md)
