# Phase 1 — Ubuntu / VirtualBox / TVM Host Environment

本文件目標是把 **x86_64 Ubuntu** 設為可編譯 ONNX 模型的 **Host**。它不是 Banana Pi 的板端安裝指南；板端另見 [banana_pi_deployment.md](banana_pi_deployment.md)。

> 來源與版本：參考 [930727fre/tvm_rvv_matmul_byoc 的 compile_tvm.md](https://github.com/930727fre/tvm_rvv_matmul_byoc/blob/main/compile_tvm.md)，該專案固定 **TVM v0.20.0、LLVM 15**，並使用 `bananapi` backend。本文件主線改為 **lewis21456/tvm-byoc-kiwipedia 自帶的 TVM 原始碼**，不應直接切換至上游 TVM v0.20.0，否則 Kiwipedia 的修改可能消失。依 Ubuntu、GCC、TVM commit 不同，套件版本可能需要固定；尚未在全新機器端到端實測。

## 0. VirtualBox（選用）

參考專案的實驗 VM 為 Ubuntu **22.04.5 LTS、6 vCPU、20 GB RAM、100 GB Disk**。這是一組歷史設定，可依實際主機資源調整；亦可直接在 Ubuntu 實體機執行，不必使用 VirtualBox。

1. 建立 Ubuntu 22.04 x86_64 虛擬機；磁碟空間預留足夠容量給 TVM、LLVM、交叉工具鏈與模型。
2. 若要主客系統共用複製貼上，在 VirtualBox 的「Devices → Shared Clipboard → Bidirectional」啟用（可能需安裝 Guest Additions）。
3. 長時間編譯時可在 Ubuntu「Settings → Power」延後或停用自動休眠，以免工作中斷。

下列命令都在 **Ubuntu Host Terminal** 執行，不是在 Banana Pi 執行。

## 1. 安裝系統工具

```bash
sudo apt update
sudo apt install -y \
  git git-lfs rsync openssh-client \
  build-essential cmake ninja-build pkg-config \
  llvm-15 llvm-15-dev llvm-15-tools clang-15 \
  python3 python3-pip python3-venv python3-dev \
  ffmpeg libsndfile1 file binutils \
  zlib1g-dev libtinfo-dev libxml2-dev \
  autoconf automake autotools-dev curl wget

git lfs install
cmake --version
llvm-config-15 --version
python3 --version
```

參考專案指定 LLVM 15；本專題 TVM fork 未公布完整的依賴鎖定檔，所以這裡採用**參考版本，而不是宣稱所有 commit 都已驗證**。若 CMake 無法找到 LLVM，編譯時可改指定 `-DUSE_LLVM="$(command -v llvm-config-15)"`，但要先確認本 fork CMake 接受這個變數。Ubuntu 套件缺失時請檢查發行版本及套件來源。

## 2. Clone 主專案並取得 ONNX（不要複製別人的整套 TVM）

```bash
cd "$HOME"
git clone https://github.com/lewis21456/tvm-byoc-kiwipedia.git
cd tvm-byoc-kiwipedia

git submodule update --init --recursive
git lfs pull

git rev-parse HEAD
git lfs ls-files
```

先確認專案確實含有自訂原始檔：

```bash
test -d tvm/python
find tvm/src -type f \( \
  -name 'kiwipedia_runtime.cc' -o \
  -name 'kiwipedia_codegen.cc' -o \
  -name 'libmatmul*.cpp' \)
```

檢查模型檔是否存在、而非 Git LFS pointer：

```bash
for m in encoder_model decoder_model decoder_with_past_model; do
  f="whisper-tiny/onnx/${m}.onnx"
  test -s "$f" || { echo "缺少 $f"; exit 1; }
  if head -c 80 "$f" | grep -aq 'git-lfs.github.com/spec/v1'; then
    echo "需要 git lfs pull: $f"
    exit 1
  fi
done
```

如需另外下載 Processor／Tokenizer，可由 [Hugging Face onnx-community/whisper-tiny](https://huggingface.co/onnx-community/whisper-tiny/tree/main) 取得模型所需的 tokenizer 設定檔。**不是隨便下載一組 `.onnx` 就能直接替換**：三個編譯腳本的 Tensor 名稱、形狀、Decoder past 介面必須對應。

完成下一步的 Python 虛擬環境後，如果 `whisper-tiny/` 缺少 Processor／Tokenizer 設定，可在 **Host x86 Ubuntu** 執行：

```bash
cd "$HOME/tvm-byoc-kiwipedia"
source .venv/bin/activate
python -m pip install huggingface_hub
python - <<'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="onnx-community/whisper-tiny",
    local_dir="whisper-tiny",
    allow_patterns=["*.json", "*.txt", "*.model"],
)
PYCODE
```

上述只下載模型設定／Tokenizer 檔案，**不會覆寫專案原有三個 ONNX**。模型的 `WhisperProcessor` 語言、任務及 Tokenizer 要與實際 ONNX 相容，下載後仍需以推論程式驗證。

## 3. 建立 Python 虛擬環境

```bash
cd "$HOME/tvm-byoc-kiwipedia"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install \
  numpy scipy onnx transformers sentencepiece soundfile librosa \
  ninja psutil cython pillow scikit-build-core pybind11 \
  packaging yt-dlp huggingface_hub

which python
```

`torch` 依模型匯出路徑或套件版本才需要；既有 ONNX 推論不應無條件安裝全套 CUDA PyTorch。若新版套件 API 不相容，應將成功運作的套件版本輸出為 `requirements-lock.txt`。

## 4. 編譯 Host TVM（Kiwipedia BYOC）

```bash
cd "$HOME/tvm-byoc-kiwipedia"
source .venv/bin/activate

cmake -S tvm -B tvm/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_LLVM=ON \
  -DUSE_KIWIPEDIA_CODEGEN=ON \
  -DUSE_KIWIPEDIA_RUNTIME=ON

# RAM 不足可改成 --parallel 2 或 --parallel 4
cmake --build tvm/build --parallel 4
ls -lh tvm/build/libtvm.so
```

如果本 fork 的 CMake 另外依 `tvm/build/config.cmake` 設定，先依該版本配置。執行 `cmake -LA -N tvm/build | grep -E 'USE_LLVM|KIWIPEDIA'` 可以看已使用的 CMake cache；如果有 **`Manually-specified variables were not used`**，不代表 BYOC 已生效，還要做第 7 步的註冊檢查。

修改 `tvm/src` 或 CMake 內的 C++ Codegen／Runtime 後，需重新編譯：

```bash
cmake --build "$HOME/tvm-byoc-kiwipedia/tvm/build" --parallel 4
```

## 5. Host x86 外部 MatMul（用來驗證 Host 安裝，並非 RVV Kernel）

```bash
cd "$HOME/tvm-byoc-kiwipedia/tvm/src/runtime/contrib/kiwipedia"
DLPACK_DIR="$HOME/tvm-byoc-kiwipedia/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include"
test -f "$DLPACK_DIR/dlpack/dlpack.h" || { echo 'DLPack header 不存在，請檢查子模組'; exit 1; }
g++ -O3 -shared -fPIC -I"$DLPACK_DIR" libmatmul.cpp -o libmatmul.so
file libmatmul.so
```

這個 `.so` 應顯示 `x86-64`。**不能複製到 Banana Pi 執行**；RVV 版本需在目標板或交叉工具鏈重新編譯。

## 6. 安裝 TVM FFI 與環境變數

```bash
cd "$HOME/tvm-byoc-kiwipedia"
python -m pip install -e ./tvm/3rdparty/tvm-ffi

cp scripts/env.example.sh scripts/env.local.sh
source scripts/env.local.sh

printf 'TVM_HOME=%s\n' "$TVM_HOME"
printf 'KIWIPEDIA_MATMUL_SO=%s\n' "$KIWIPEDIA_MATMUL_SO"
```

> **注意**：最後兩行 `cp`／`source` 的前提是已把本文件包 `scripts/` 合併進原始 Repo。若尚未合併，請依 [README Quick Start 第 6 步](../README.md#6-設定環境變數並驗證-tvm) 建立 `env_tvm.sh`。不應將個人 `env.local.sh` 上傳 GitHub。

必需的 Host 變數包括：

| 變數 | Host 意義 |
| --- | --- |
| `PROJECT_ROOT` | `$HOME/tvm-byoc-kiwipedia` |
| `TVM_HOME` | `$PROJECT_ROOT/tvm` |
| `PYTHONPATH` | `$TVM_HOME/python` 及本 fork 的 FFI Python 路徑 |
| `LD_LIBRARY_PATH` | **x86 Host** 的 `$TVM_HOME/build` |
| `KIWIPEDIA_MATMUL_SO` | **x86 Host** `libmatmul.so` 絕對位置 |

## 7. 驗證 Host TVM／BYOC

```bash
cd "$HOME/tvm-byoc-kiwipedia"
source .venv/bin/activate
source scripts/env.local.sh  # 如果沒有 scripts/，改為 source env_tvm.sh

python - <<'PY'
import tvm
print('Loaded TVM:', tvm.__file__)
for name in ['target.build.llvm', 'runtime.kiwipedia_runtime_create']:
    fn = tvm.get_global_func(name, True)
    print(name, 'OK' if fn is not None else 'NOT REGISTERED')
    assert fn is not None, name
PY
```

這裡的檢查只代表 Host 的相關 TVM 功能已註冊，**不代表 RVV 編譯或板端推論已完成**。

## 8. 主機端先跑一次模型（選用，但強烈建議）

請依 [README Quick Start 第 7–10 步](../README.md#7-編譯-whisper-tiny-的三個-onnx-模型) 編譯及執行 Host x86 版本，先排除 ONNX／Processor／Runtime 問題。完成後再進入下一階段：[RISC-V 工具鏈](compile_riscv_toolchain.md)。

## 關於參考專案使用 TVM v0.20.0

[930727fre 的原始 `compile_tvm.md`](https://github.com/930727fre/tvm_rvv_matmul_byoc/blob/main/compile_tvm.md) 採 `git clone -b v0.20.0 --recursive https://github.com/apache/tvm tvm`。**那是 `bananapi` 舊版本重現路徑，不是本專題的 Kiwipedia 原始碼來源**。如果要獨立重現舊專案，請另建資料夾，並依其 C++ patch／CMake 模組建置；兩套 TVM 不要共用 `PYTHONPATH`、`build/` 或 `.so`。

[← README](../README.md) · [下一步：RISC-V Toolchain →](compile_riscv_toolchain.md)
