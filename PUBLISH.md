# Publish the Project to GitHub

本頁提供**已有 GitHub 帳號與 Repository** 的情況下，如何把整理好的專案文件加入現有 Repo，而不是重新建立整份 TVM 專案。

## 1. 整合文件（在你的現有 repo 中）

將本範本中的下列檔案複製到既有專題 repo 根目錄：

```text
README.md
.gitignore
docs/BUILD.md
docs/compile_tvm.md
docs/compile_riscv_toolchain.md
docs/set_up_bananapi_runtime_and_codegen.md
docs/banana_pi_deployment.md
docs/REFERENCE_COMPARISON.md
docs/STRUCTURE.md
docs/PUBLISH.md
scripts/env.example.sh
scripts/env.bananapi.example.sh
scripts/check_host.sh
scripts/check_environment.sh
```

如果原本已有 `README.md` 或 `.gitignore`，請**先檢查差異並合併內容**，不要直接覆寫。若你的模型編譯腳本已存在其他路徑，也不必為了文件架構搬動原始碼，只要修正 README／BUILD 中的相對位置即可。

## 2. 檢查 Git 狀態

```bash
cd <EXISTING_REPOSITORY>
git status
git branch --show-current
git remote -v
```

如果已在正確的 GitHub repo 中，**不需要執行 `git init`、也不要重新設定 remote**。

## 3. 上傳公開文件

```bash
# 確認 README、文件與腳本已加入此 Repo
git add README.md .gitignore docs/ scripts/env.example.sh scripts/env.bananapi.example.sh scripts/check_host.sh scripts/check_environment.sh

# 檢查要上傳的內容，尤其不要包含 env.local.sh、模型權重及秘密資訊
git diff --cached --stat
git diff --cached

git commit -m "docs: add BYOC RVV setup, build and deployment guide"

git push origin <YOUR_BRANCH>
```

若專案使用 `main` 分支，可將 `<YOUR_BRANCH>` 改為 `main`。如未設定上游分支，可用：

```bash
git push -u origin <YOUR_BRANCH>
```

## 4. 正式公開前

- [ ] README 所列腳本名稱、相對路徑與 GitHub 實際檔案一致
- [ ] 從**乾淨的 clone** 至少測試過環境安裝與 Host TVM 編譯
- [ ] 交叉編譯命令已由 Placeholder 換成實際成功紀錄
- [ ] 引用學長原有 BYOC／Kernel 程式碼的來源及授權已註明
- [ ] TVM fork 的修改、commit、子模組來源均可取得
- [ ] ONNX 模型／權重發布方式符合模型授權與儲存限制
- [ ] `.gitignore` 與 `git status` 確認不含個人路徑、金鑰、秘密資訊
- [ ] Benchmark 的輸入、裝置、初始化、測試次數、正確性檢查均有描述

> 若目前只有報告與本文件範本，也可以先建立一個**documentation-only** Repository，但請清楚標示「原始碼／模型檔案尚未公開」，不要聲稱已達到完整可重現狀態。

[← 返回 README](../README.md)

> 本 Quick Start 第 6 步會產生個人用的 `env_tvm.sh`；請將它加入 `.gitignore`，不要提交到 GitHub。
