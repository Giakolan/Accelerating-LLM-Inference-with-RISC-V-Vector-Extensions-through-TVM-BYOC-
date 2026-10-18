# Phase 3 — Deploy Whisper-tiny to Banana Pi BPI-F3

本篇接續 [RISC-V 模型交叉編譯](set_up_bananapi_runtime_and_codegen.md)。**所有三個模型 `.so` 必須先通過 `file`／`readelf` 的 `RISC-V` 檢查**；x86 `.so` 無法直接在 Banana Pi 執行。

來源：參考 [930727fre/tvm_rvv_matmul_byoc README](https://github.com/930727fre/tvm_rvv_matmul_byoc) 的 **Bianbu 2.2、TVM v0.20.0、rsync、native RVV Kernel + dlopen** 部署思路；此處保留本專題的 **Kiwipedia 命名與路徑**。本專題板端 TVM 及套件的確切版本仍需由實際成功環境補上，以下不能視為已通過本機實測。

## 0. 先備條件

- Banana Pi BPI-F3 可開機、已連線，並能透過 SSH 登入。
- 板端執行 RISC-V Linux（參考專案使用 Bianbu 2.2 `riscv64`；版本不同需依實際套件調整）。
- x86 Host 已完成三個 **RISC-V** `.so`，且 Model/Tokenizer 資產可用。
- Kiwipedia 的板端 TVM Runtime 版本與模型匯出使用的 TVM API／ABI 相容。

## 1. 在 Host 設定連線資訊，測試 SSH

```bash
# 在 x86 Ubuntu Host 執行（請自行填入實際帳號與 IP）
export BOARD_USER='YOUR_BOARD_USER'
export BOARD_IP='YOUR_BOARD_IP'
export BOARD="$BOARD_USER@$BOARD_IP"

ssh "$BOARD" 'uname -m; cat /etc/os-release | head -6'
```

請確認 `uname -m` 顯示 `riscv64`。不要把原參考作者的帳號或其個人 IP 直接寫死在公開 README。

## 2. Host 端確認模型格式

```bash
cd "$HOME/tvm-byoc-kiwipedia/whisper-tiny/onnx"
for f in encoder_model.so decoder_model.so decoder_with_past_model.so; do
  test -s "$f" || { echo "缺少 $f"; exit 1; }
  echo "== $f =="
  file "$f"
  readelf -h "$f" | grep 'Machine:'
done
```

若顯示 `Advanced Micro Devices X86-64`，請**停止部署**，回到 [Codegen / Cross-compile](set_up_bananapi_runtime_and_codegen.md) 修正 `target` 與 `fcompile`。

## 3. 用 rsync 傳模型與推論程式

參考專案範例：`rsync -avz -e ssh ./*.so <board-user>@<ip>:~/whisper-tiny/onnx/`。本專案位於 `~/tvm-byoc-kiwipedia/whisper-tiny/onnx/`，因此統一路徑如下：

```bash
# x86 Ubuntu Host
cd "$HOME/tvm-byoc-kiwipedia"

ssh "$BOARD" \
  'mkdir -p ~/tvm-byoc-kiwipedia/whisper-tiny/onnx'

rsync -avz -e ssh whisper-tiny/onnx/*_model.so \
  "$BOARD:~/tvm-byoc-kiwipedia/whisper-tiny/onnx/"

rsync -avz -e ssh whisper-tiny/inference.py \
  "$BOARD:~/tvm-byoc-kiwipedia/whisper-tiny/"
```

若最終專題改用 `inference_long.py` 或 VM 常駐程式，應將真正使用的推論腳本一起傳過去，不要假設這份公開 `inference.py` 內已含 O5 的長音訊流程。

板端還需要**與 ONNX 匹配**的 Processor／Tokenizer 檔案（如 `tokenizer.json`、`vocab.json`、`merges.txt`、`preprocessor_config.json` 等；實際檔名取決於模型版本）。可從 Host 將已存在的檔案複製：

```bash
# 從 Host whisper-tiny 根目錄只同步需要的模型文字設定檔
cd "$HOME/tvm-byoc-kiwipedia/whisper-tiny"
rsync -avz -e ssh --prune-empty-dirs \
  --include='/*.json' --include='/*.txt' --include='/*.model' \
  --exclude='/*' ./ \
  "$BOARD:~/tvm-byoc-kiwipedia/whisper-tiny/"
```

若檔案不存在，請依 [Hugging Face 模型倉庫](https://huggingface.co/onnx-community/whisper-tiny/tree/main) 下載匹配版本；不要將不相容的 ONNX / Processor 混用。

## 4. 編譯板端 RVV MatMul Kernel

參考專案**在 Banana Pi 本機**以 `g++ -march=rv64gcv -mabi=lp64d` 編譯 `libmatmul_rvv.cpp`，並供 Runtime `dlopen()`。本專案對應檔案是 `tvm/src/runtime/contrib/kiwipedia/libmatmul_rvv.cpp`（以最終版為準）。

先將必要的 TVM fork 原始碼同步到板端，或在板端 Clone **相同版本**的 repo；下面採取 Clone 範例：

```bash
# 下列命令開始在 Banana Pi Terminal 執行
sudo apt update
sudo apt install -y git git-lfs build-essential cmake \
  python3 python3-venv python3-pip \
  ffmpeg libsndfile1 file

git lfs install
cd "$HOME"
# 如果第 3 步已建立此目錄，請勿重新 clone 到同一路徑。
# 可先 git clone 到別處，或在 HOST 端 rsync 原始碼進板端。
```

**建議做法：** 在 Host 另外開一個 terminal，以 rsync 將 **同一版本** 的 TVM 原始碼同步到板端（排除 x86 build 及虛擬環境）：

```bash
# x86 Ubuntu Host（需要先設定 BOARD）
cd "$HOME/tvm-byoc-kiwipedia"
rsync -avz -e ssh \
  --exclude='/build/' --exclude='/.venv/' --exclude='*.so' \
  tvm/ "$BOARD:~/tvm-byoc-kiwipedia/tvm/"
```

確認 TVM 第三方子模組的標頭及需要的原始檔都已傳到板端；也可在板端另 clone **同一 Git commit** 再 `git submodule update --init --recursive`。這一步**不能用 Host x86 `build/` 成果代替**。

假設板端已具備 `~/tvm-byoc-kiwipedia/tvm/...` 的原始碼，執行：

```bash
# Banana Pi
export PROJECT_ROOT="$HOME/tvm-byoc-kiwipedia"
export TVM_HOME="$PROJECT_ROOT/tvm"
cd "$TVM_HOME/src/runtime/contrib/kiwipedia"

DLPACK_DIR="$TVM_HOME/3rdparty/tvm-ffi/3rdparty/dlpack/include"
test -f "$DLPACK_DIR/dlpack/dlpack.h" || {
  echo '請確認 TVM 子模組與 DLPack 標頭路徑'; exit 1;
}

g++ -std=c++17 -O3 -shared -fPIC \
  -march=rv64gcv -mabi=lp64d \
  -I"$DLPACK_DIR" \
  libmatmul_rvv.cpp -o libmatmul.so

file libmatmul.so
```

> 如 RVV Kernel 需要其他 headers（例如 `common.h`）、OpenMP 或外部物件檔，需依**最終版程式**增加 `-I`、`-fopenmp`、連結選項。參考舊 repo 把 DLPack 寫成 `3 rdparty`（多了空格），本文件已更正，且按此 fork 的 tvm-ffi 目錄布局提供檢查。

## 5. 建置 Banana Pi 自身的 TVM Runtime／Python 模組

**這步不可省略，也不能把 x86 的 `libtvm.so` 直接 rsync 到板端。** 參考專案的做法是在 Banana Pi 端安裝相容 TVM；本專題要以**相同 Kiwipedia fork**編譯出對應的 RISC-V Runtime。完整相容版本需實測。

```bash
# Banana Pi
cd "$HOME/tvm-byoc-kiwipedia"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# 若此 fork 需要 tvm_ffi：先確認能在 riscv64 系統建置及匯入
python -m pip install -e ./tvm/3rdparty/tvm-ffi

# 板端原生編譯，減少併發避免記憶體不足
cmake -S tvm -B tvm/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_LLVM=OFF \
  -DUSE_KIWIPEDIA_CODEGEN=OFF \
  -DUSE_KIWIPEDIA_RUNTIME=ON
cmake --build tvm/build --parallel 2
```

**注意：** 上面的「`USE_LLVM=OFF`、不建置 Codegen，只建置 Runtime」是合理的**板端配置範例，並非已證實與本 fork 完全相容的設定**。如果 CMake 不支援 `USE_KIWIPEDIA_RUNTIME` 或 Runtime 依賴 LLVM／其他模組，需依本 fork 目前的 CMake 實作調整。板端也可能選擇只建 `tvm_runtime`，但須核對 `kiwipedia_runtime.cc` 及 VM 所依賴的符號是否都包含在內。請以實際可正常載入模型並推論作為完成標準。

板端 Python 推論所需的 `transformers`、`soundfile`、`numpy` 等套件，可能因 riscv64 wheel 缺失而需要使用系統套件、原生編譯或版本調整。**不建議盲目在板端安裝 Host 的 Python wheel**：

```bash
# Banana Pi：依可用 wheel 與相容版本安裝
python -m pip install numpy soundfile transformers sentencepiece
```

如果安裝失敗，應先診斷套件、ABI 與 Python 版本問題，而不是把 Host `.venv` 複製到板端。

## 6. 板端環境變數

如果已把本文件包的範本檔加入 repo，請執行：

```bash
cd "$HOME/tvm-byoc-kiwipedia"
cp scripts/env.bananapi.example.sh scripts/env.bananapi.local.sh
source .venv/bin/activate
source scripts/env.bananapi.local.sh

python -c 'import tvm; print(tvm.__file__)'
file "$KIWIPEDIA_MATMUL_SO"
```

或直接在板端設定：

```bash
export PROJECT_ROOT="$HOME/tvm-byoc-kiwipedia"
export TVM_HOME="$PROJECT_ROOT/tvm"
export PYTHONPATH="$TVM_HOME/python:$TVM_HOME/3rdparty/tvm-ffi/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$TVM_HOME/build:$TVM_HOME/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export KIWIPEDIA_MATMUL_SO="$TVM_HOME/src/runtime/contrib/kiwipedia/libmatmul.so"
```

板端 `LD_LIBRARY_PATH` 必須指向**板端自己編譯出的 RISC-V TVM**，`KIWIPEDIA_MATMUL_SO` 必須指向 **RISC-V RVV Kernel**，兩者都不能使用 Host 路徑或 Host `.so`。

## 7. 準備音檔、檢查依賴、執行推論

```bash
# Banana Pi
cd "$HOME/tvm-byoc-kiwipedia/whisper-tiny"
ls -lh onnx/*_model.so
file onnx/encoder_model.so
file "$KIWIPEDIA_MATMUL_SO"

# 將合法持有的原始音檔轉成 16 kHz mono WAV；請替換輸入路徑
ffmpeg -y -i /path/to/your_audio.wav -ac 1 -ar 16000 -c:a pcm_s16le audio.wav

# 啟動實際版本的 Whisper inference
python inference.py
```

此 `inference.py` 的相對路徑要求以原始程式為準；本專題已知公開版使用 `whisper-tiny/` 為工作目錄。要重現 **O4／O5（15/30 秒 bucket、VM 常駐、長音訊）**，必須改用報告最終版相應的推論腳本與模型 `.so`，不可拿初始公開版推論時間和 O5 數字直接比較。

## 8. 每次開機後快速執行

```bash
cd "$HOME/tvm-byoc-kiwipedia"
source .venv/bin/activate
source scripts/env.bananapi.local.sh
cd whisper-tiny
python inference.py
```

## 9. 常見問題

| 症狀 | 排除方式 |
| --- | --- |
| `Exec format error` | `.so` 架構錯誤，使用 `file` / `readelf -h` 查看 `Machine: RISC-V` |
| `GLIBC_... not found` | 交叉編譯的 sysroot/glibc 與板端版本不相容 |
| `libtvm.so` 不存在或載入失敗 | 板端 TVM Runtime 尚未正確編譯；檢查 `LD_LIBRARY_PATH` |
| `dlopen libmatmul.so failed` | 檢查 `KIWIPEDIA_MATMUL_SO`，並用 `ldd` 查看必要依賴 |
| `Illegal instruction` | 目標 ISA 和硬體不相容，或使用不支援的 RVV 版本 |
| `cannot import tvm_ffi` | 板端 Python venv、FFI 版本、套件 native build 未完成 |
| `WhisperProcessor` 載入錯誤 | Processor／Tokenizer 檔案不齊，或與模型版本不匹配 |
| 模型可跑但 RVV 沒加速 | 驗證 Runtime 有呼叫外部 Kernel；檢查 `objdump`／Profiling |
| 動態 KV cache 生成錯誤 | 核對 Dynamic Shape Pass、Runtime Tensor shape 與 `EntryID` 映射 |

**性能與正確性應分開看：** QEMU 主要確認 ELF、指令及輸出正確；Whisper `jfk.flac` 的端到端時間需要以 Banana Pi 實機測試，且固定模型版本、音檔與計時條件。

[← README](../README.md) · [參考版本差異 →](REFERENCE_COMPARISON.md)
