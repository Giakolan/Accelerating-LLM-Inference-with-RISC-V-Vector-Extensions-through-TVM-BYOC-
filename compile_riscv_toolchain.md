# Phase 2A — Install RISC-V GNU Toolchain (Linux + RVV)

本步在 **x86 Ubuntu Host** 執行。目標是在 Host 產生 `riscv64-unknown-linux-gnu-g++`，供 RISC-V Linux 的 `.so` 交叉編譯；不是裸機 `riscv64-unknown-elf-g++`。

**參考來源：** [930727fre/compile_riscv_toolchain.md](https://github.com/930727fre/tvm_rvv_matmul_byoc/blob/main/compile_riscv_toolchain.md)。原流程包含 Newlib（`make`）及 Linux（`make linux`）兩類工具鏈；本專題 Banana Pi 執行 Linux 並使用共享函式庫，故以 **`make linux` 為必要**，Newlib 為選用。

## 1. 安裝建置依賴

```bash
sudo apt update
sudo apt install -y \
  autoconf automake autotools-dev curl python3 python3-pip python3-tomli \
  libmpc-dev libmpfr-dev libgmp-dev gawk build-essential \
  bison flex texinfo gperf libtool patchutils bc \
  zlib1g-dev libexpat-dev ninja-build git cmake \
  libglib2.0-dev libslirp-dev
```

## 2. 取得原始碼（含子模組）

```bash
cd "$HOME"
git clone --recursive https://github.com/riscv-collab/riscv-gnu-toolchain.git riscv-gnu-toolchain
cd riscv-gnu-toolchain
git submodule update --init --recursive

git rev-parse HEAD
```

參考 repo 使用 `https://github.com/riscv/riscv-gnu-toolchain`；它的維護位置及子模組可能隨時間改變，若上方 URL 不可用請依官方當前 repo 說明取得相容的來源。為了重現，請記錄 commit。

## 3. 配置與編譯 Linux Toolchain

```bash
cd "$HOME/riscv-gnu-toolchain"
mkdir -p build-linux
cd build-linux

../configure \
  --prefix="$HOME/opt/riscv" \
  --with-arch=rv64gcv \
  --with-abi=lp64d

# 大型建置，可能耗時很久；不要在記憶體不足時使用全部核心
make linux -j4
```

> 原參考稿有 `mkdir build_riscv && cd build_risc` 的目錄拼字不一致，不能原樣執行。本指南已改成 `build-linux`。它使用 `/opt/riscv` 並搭配 `sudo make linux`；此處改為安裝在使用者家目錄，通常不必 sudo。實際 GCC／glibc 工具鏈仍需與 Banana Pi 系統 ABI 相容。

`--with-arch=rv64gcv` 可作為 RISC-V RVV 工具鏈範例；如果 GCC 對 ISA／ABI 或目標 sysroot 有限制，須依實際裝置支援情況調整。此旗標**並不保證** TVM 或模型程式自動使用 RVV；RVV Kernel 的編譯還要另外設定。

## 4. 設定 PATH 與交叉編譯環境變數

如果已合併本文件包，可以直接 `source scripts/env.local.sh`（內容先依實際安裝位置調整）。否則先手動設定：

```bash
export RISCV_TOOLCHAIN_ROOT="$HOME/opt/riscv"
export PATH="$RISCV_TOOLCHAIN_ROOT/bin:$PATH"
export RISCV_CXX="$RISCV_TOOLCHAIN_ROOT/bin/riscv64-unknown-linux-gnu-g++"
export RISCV_SYSROOT="$("$RISCV_CXX" -print-sysroot)"

command -v riscv64-unknown-linux-gnu-g++
"$RISCV_CXX" --version
"$RISCV_CXX" -print-sysroot
```

不建議將 `RISCV_SYSROOT` 永遠寫成 `~/opt/riscv/sysroot` 而不檢查；以編譯器查詢結果為準。**RISC-V Linux 工具鏈 sysroot 與板端 Bianbu 的 glibc/loader 版本可能不同**，若執行時出現 `GLIBC_... not found`，需要改用與目標系統相容的 sysroot 或在板端 native build。

## 5. 編譯最小 RVV 測試程式並檢查 ELF

```bash
cat >/tmp/hello_riscv.cc <<'CPP'
#include <iostream>
int main() { std::cout << "Hello RISC-V Linux\n"; return 0; }
CPP

"$RISCV_CXX" -O2 -march=rv64gcv -mabi=lp64d \
  /tmp/hello_riscv.cc -o /tmp/hello_riscv

file /tmp/hello_riscv
readelf -h /tmp/hello_riscv | grep -E 'Class:|Machine:'
```

預期 `Machine: RISC-V`。這個測試只證明交叉編譯器可輸出 RISC-V Linux ELF，**不證明向量指令或端到端模型已工作**。

## 6. QEMU user-mode：可選正確性驗證

如果系統已提供支援 RVV 的 `qemu-riscv64`：

```bash
sudo apt install -y qemu-user
command -v qemu-riscv64

# 確認 QEMU 版本支援指定 CPU/向量選項
qemu-riscv64 -cpu help

qemu-riscv64 -L "$RISCV_SYSROOT" -cpu max /tmp/hello_riscv
```

如果 Ubuntu apt 的 QEMU 版本無法支援所需 RVV v1.0 功能，則需要另外建置支援 RVV 的 QEMU。既有研究記錄使用 `$HOME/qemu-rvv/bin/qemu-riscv64`，但其原始碼版本與完整建置參數未由報告確認；不要憑空聲稱已重現。

## 7. 常見問題

| 問題 | 檢查方式 |
| --- | --- |
| 找不到 `riscv64-unknown-linux-gnu-g++` | 檢查 `make linux` 是否成功、`PATH` 是否包含 `$HOME/opt/riscv/bin` |
| `.so` 顯示 `x86-64` | 實際仍用了 Host `g++`，應明確呼叫 `$RISCV_CXX` |
| `riscv_vector.h` 找不到 | 使用過舊編譯器或無 RVV Intrinsics，需檢查交叉工具鏈版本 |
| QEMU 出現 `Could not open ... ld-linux...` | `-L` 與 sysroot 未對應，或 glibc 動態載入器不存在 |
| Bianbu 上找不到 `GLIBC_x.y` | Host 工具鏈的 glibc 高於板端，改用相容 sysroot 或板端編譯 |
| 交叉工具鏈 OOM / build 中斷 | 使用較少執行緒 (`make linux -j2`)、增加 RAM/Swap |

## 下一步

請先閱讀 [Kiwipedia Codegen／Runtime 及 RISC-V 模型交叉編譯](set_up_bananapi_runtime_and_codegen.md)。

[← README](../README.md) · [下一步：Codegen / Runtime →](set_up_bananapi_runtime_and_codegen.md)
