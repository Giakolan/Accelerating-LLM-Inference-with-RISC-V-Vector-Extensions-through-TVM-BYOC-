# Reference Comparison — Senior `bananapi` vs. Our `kiwipedia`

這份對照表協助讀者理解，哪些步驟來自學長公開版本，哪些屬於本專題 Kiwipedia 的修改與後續研究成果。不能因為兩個專案都叫 TVM BYOC / Whisper / RVV，就把 Python 腳本、CMake 檔或 Runtime 原始碼互相覆蓋。

## Sources

- [Reference repository: 930727fre/tvm_rvv_matmul_byoc](https://github.com/930727fre/tvm_rvv_matmul_byoc)
- [Reference TVM compilation](https://github.com/930727fre/tvm_rvv_matmul_byoc/blob/main/compile_tvm.md)
- [Reference RISC-V toolchain](https://github.com/930727fre/tvm_rvv_matmul_byoc/blob/main/compile_riscv_toolchain.md)
- [Reference Codegen / Runtime setup](https://github.com/930727fre/tvm_rvv_matmul_byoc/blob/main/set_up_bananapi_rutime_and_codegen.md)（原 repo 檔名包含 `rutime` 拼字）
- [Current public Kiwipedia project](https://github.com/lewis21456/tvm-byoc-kiwipedia)
- 本專題報告 *Accelerating LLM Inference with RISC-V Vector Extensions through TVM BYOC*，特別是 Dynamic Shape、RVV MatMul、Shape Bucket、研究方法及效能結果章節。

## Version / architecture boundaries

| 項目 | 930727fre 參考版 | 本專題 Kiwipedia / 報告 |
| --- | --- | --- |
| Backend 名稱 | `bananapi` | `kiwipedia` |
| Host TVM | 參考明確固定 TVM v0.20.0 | 自訂 fork；公開倉庫含 `tvm/`，最終 commit／相容 TVM 版本需核對 |
| Compiler | LLVM 15 / RISC-V GNU Linux toolchain | LLVM + RISC-V GNU；實際版號需鎖定 |
| Board | Banana Pi BPI-F3，參考用 Bianbu 2.2 | Banana Pi BPI-F3；報告以 SpacemiT K1 為目標 |
| RVV kernel | `libmatmul_rvv.cpp`，板端 native 編成 `.so` | RVV MatMul + Shape-tuned Kernel；具體原始碼版本需補 |
| Dynamic decoder | 參考 `compile_decoder_with_past.py` 的 BYOC `FuseOpsByPattern/RunCodegen` 為註解 | 報告包含 `RemoveKiwipediaShapeParams`、Runtime Tensor shape 修正 |
| 15 秒 Encoder / M=750 | 參考版固定 30 秒輸入 | 報告 O4 增加 15 秒 Shape Bucket |
| VM 重用與長音訊 | 參考版未證實具備 O5 流程 | 報告 O5 具備 VM 常駐、Chunk / overlap |
| 公開 x86 Quick Start | 參考 repo `tutorial/` 與 RISC-V 主流程分離 | `lewis21456` 公開三支腳本目前以 Host `target="llvm"` 為主 |

## Why the reference is useful

- 清楚列出 x86 Host、RISC-V Compiler、板端 native Runtime 三個不同的執行環境。
- 展示 `ex.export_library(..., fcompile=riscv_fcompile)` 透過 RISC-V GNU `g++` 交叉連結 ELF。
- 提供 TVM 需要修改的 CMake、Codegen、Runtime 整合點。
- 說明 `dlopen()` 分離 Host TVM 與板端 RVV `libmatmul.so` 的動機。
- 示範 `rsync` 搬移三個 `.so`，並在板端載入與執行。

## Corrections / cautions when adapting the reference

1. 原 `compile_riscv_toolchain.md` 中 `mkdir build_riscv && cd build_risc` 有拼字不一致；本文件統一 `build-linux`。
2. 原 README 的 DLPack include 範例出現 `3 rdparty`，不可直接照抄；本專題確認 header 位置後再編譯。
3. 原 README 提供的 x86 `g++` 編譯指令僅示意；真正 `riscv_vector.h` 的 RVV kernel 需要支援 RVV 的 compiler 和相容目標架構。
4. 原參考使用作者自己的 `/home/<user>` 路徑及板端帳號，正式教學改成 `$HOME` 和 `BOARD_USER`／`BOARD_IP`。
5. 本專題 Kiwipedia 的 `libmatmul.so` 由 `KIWIPEDIA_MATMUL_SO` 設定；不能擅自用 `bananapi` 的 Runtime 名稱替換。
6. 交叉工具鏈可生成 `riscv64` ELF，**但仍須檢查板端 loader、glibc、TVM FFI ABI 及 Runtime 相容**。
7. 原參考程式中的 `decoder_with_past` RunCodegen 註解，**不能推導本專題最終 Decoder 動態 shape 已可由該腳本直接重現**。

這份文件的目標是提供**可追溯的技術脈絡**：參考成果作為起點，報告後期的動態形狀與 shape-aware kernel 屬於本專題的延伸；請保留清楚的來源與版本標示。

[← README](../README.md)
