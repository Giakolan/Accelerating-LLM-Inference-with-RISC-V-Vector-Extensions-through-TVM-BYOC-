# Repository Layout & Source File Guide

本頁區分三種資訊：**已在既有開發紀錄中出現的檔案**、**專題報告確認的元件**，以及**方便 GitHub 訪客閱讀而建議的公開目錄**。

## 1. 已知實際原始檔（既有開發紀錄）

以下相對路徑曾由先前終端機 `find` 輸出或直接操作紀錄確認；若已重構 repo，請以新版 Git tree 為準。

```text
tvm/
└── src/
    └── runtime/
        └── contrib/
            └── kiwipedia/
                ├── kiwipedia_runtime.cc  # BYOC Runtime：張量取用、執行路徑
                ├── libmatmul.cpp         # 原有／Baseline MatMul 實作
                └── libmatmul_rvv.cpp     # RVV MatMul / Shape-aware Kernel
```

另外，部分名稱來自既有原始碼討論或報告；**本次已與目前公開 GitHub 版本確認三支模型編譯腳本及推論程式路徑**。

| 檔案 / 元件 | 功能 | 位置是否確認 |
| --- | --- | --- |
| `kiwipedia_codegen.cc` | BYOC 外部 Codegen，產生／連結相關 Kernel | **檔名已知，位置待查** |
| `RemoveKiwipediaShapeParams` | 刪除多餘的 `R.Shape` 參數，修正呼叫端 | **報告確認 Pass，位置待查** |
| `compile_encoder.py` | 匯入 ONNX、編譯 Whisper Encoder | `whisper-tiny/onnx/compile_encoder.py`（公開版：x86 LLVM） |
| `compile_decoder.py` | 編譯 Decoder Prefill | `whisper-tiny/onnx/compile_decoder.py`（公開版：x86 LLVM） |
| `compile_decoder_with_past.py` | 編譯處理 KV cache 的逐 token Decoder | `whisper-tiny/onnx/compile_decoder_with_past.py`（公開版：x86 LLVM；BYOC Codegen 為註解） |
| `inference.py` | 模型模組載入與推論 | `whisper-tiny/inference.py`（目前使用 `./onnx/*.so`） |
| `kv_cache_kernel.cc` | 先前研究的 KV cache 運算實驗 | **曾討論，不代表最終版必需** |
| `benchmark_gelu_qemu.cpp`, `gelu_rvv.cpp` | GELU RVV 正確性／效能實驗 | **既有操作紀錄，位置待查** |

**驗證命令：**

```bash
# 在專案根目錄執行
find . -type f \( \
  -name 'kiwipedia_codegen.cc' -o \
  -name 'kiwipedia_runtime.cc' -o \
  -name 'libmatmul*.cpp' -o \
  -name 'compile_*.py' -o \
  -name 'inference.py' -o \
  -name '*gelu*' \
\) | sort

grep -RIn 'RemoveKiwipediaShapeParams' tvm/src || true
```

## 1.1 參考專案的檔案對照（不要直接互相覆蓋）

由 [930727fre/tvm_rvv_matmul_byoc](https://github.com/930727fre/tvm_rvv_matmul_byoc) 確認，其 `bananapi_codegen.cc`、`bananapi_runtime.cc`、`bananapi.cmake`、`compile_*.py`、`libmatmul_rvv.cpp` 與相關 CMake 修改都是舊版 `bananapi` backend 的實作。本專題使用 `kiwipedia`，外部 Codegen 與 Runtime 名稱及註冊方式不同。見 [REFERENCE_COMPARISON.md](REFERENCE_COMPARISON.md)。

以下新增文件是本次公開教學的**建議路徑**：

```text
docs/
├── compile_tvm.md
├── compile_riscv_toolchain.md
├── set_up_bananapi_runtime_and_codegen.md
├── banana_pi_deployment.md
└── REFERENCE_COMPARISON.md
scripts/
├── env.example.sh
└── env.bananapi.example.sh
```

## 2. 建議的 GitHub 專案配置（示意，非現況）

```text
tvm-byoc-kiwipedia/
├── README.md
├── .gitignore
├── docs/
│   ├── BUILD.md
│   ├── compile_tvm.md
│   ├── compile_riscv_toolchain.md
│   ├── set_up_bananapi_runtime_and_codegen.md
│   ├── banana_pi_deployment.md
│   ├── REFERENCE_COMPARISON.md
│   ├── PUBLISH.md
│   └── STRUCTURE.md
├── scripts/
│   ├── env.example.sh
│   ├── env.bananapi.example.sh
│   ├── check_host.sh
│   ├── check_environment.sh
│   ├── compile_encoder.py             # 待納入／或改為連到現有路徑
│   ├── compile_decoder.py             # 待納入／或改為連到現有路徑
│   ├── compile_decoder_with_past.py   # 待納入／或改為連到現有路徑
│   └── inference.py                   # 待納入／或改為連到現有路徑
├── tvm/
│   ├── src/runtime/contrib/kiwipedia/
│   │   ├── libmatmul.cpp
│   │   ├── libmatmul_rvv.cpp
│   │   └── kiwipedia_runtime.cc
│   └── ...                             # 其餘 TVM + BYOC Codegen / Relax Pass
├── whisper-tiny/                         # 目前公開版已存在此路徑
│   ├── inference.py
│   └── onnx/                           # Encoder / Decoder / With Past + ONNX 模型
├── models/                              # 僅建議另建，用於新增模型／實驗版本
│   └── pythia-70m/
├── benchmarks/
│   ├── matmul/
│   ├── attention/
│   └── activation/
├── tests/                               # Correctness / compare-to-reference
└── artifacts/                           # 生成的 .so、Profile、結果，不直接追蹤
```

這份範本下載後**只包含 README、docs、env 範本與檢查腳本**；以上 `models`、`tvm`、`benchmarks`、`tests` 需要由實際專題原始碼補齊。若既有程式的相對路徑已被其他程式引用，應**保留路徑**並只更新文件，不必為了美觀任意搬動。

## 3. 模組對照與執行順序

| 模組 | 輸入 | 核心運算／議題 | 推論用途 |
| --- | --- | --- | --- |
| Whisper Encoder | Mel Spectrogram；15/30 秒 Shape | Conv1D、`M=750/1500` MatMul | 取得音訊特徵 |
| Decoder Prefill | 起始 Token + Encoder 狀態 | MatMul、Cross Attention | 初始化本次生成 |
| Decoder with Past | 目前 Token + Past KV cache | Dynamic Shape、部分 `M=1` MatMul | 自回歸逐 token 推論 |
| Pythia-70M | 固定 16-token 輸入 | `M=16` Shape-aware / Auto-vectorized / RVV | 一次完整 Forward Pass |

> 研究報告記載：每個新的音訊 Chunk 必須初始化起始 Token 與 KV cache；跨 Chunk 重用的是 Processor、Tokenizer、已載入的模型模組與 VM，而非上一段音訊的 KV cache。

## 4. 建議如何管理修改後的 TVM

**建議選擇其中一種方式**，務必公開可重現的來源：

- **Fork：** 將 TVM + BYOC 改動提交至有版本記錄的 fork，主專案說明所依賴的 commit。
- **Submodule：** 主 repo 中的 `tvm/` 指向**已經包含自訂改動**的 TVM fork 與確切 commit（不是未改動的官方 upstream）。
- **Patches：** 記錄 upstream commit，並提交 `patches/`、套用腳本、預期 SHA 與建置說明。

> 不建議只發布編譯好的 `.so` 而不提供 Codegen / Runtime 原始碼，因為 Dynamic Shape 修正與 Shape Dispatcher 才是本研究的重要重現條件。

## 5. 公開前應清除或排除的內容

- `.venv/`、`build/`、`build-riscv/`、`__pycache__/`、`*.o`、測試日誌與大型生成 `.so`。
- 不應提交含本機私有路徑的 `scripts/env.local.sh`。
- 對大型模型權重與 ONNX，依授權選用下載腳本或 Git LFS，並在 README 註明版本與來源。
- 如需公開完整測試結果，建議另存 `results/*.csv`、硬體環境說明與正確性測試方法。

[← 返回 README](../README.md)
