# Git 管理与回滚

## 初始化

```bash
bash scripts/init_git.sh
```

默认提交：

- `README.md`
- `docs/`
- `configs/`
- `records/` 模板
- `schemas/`
- `scripts/`
- `src/`
- `pyproject.toml`

不会提交：

- `external/`
- `data/`
- `runs/`
- `logs/`
- 模型权重

## 每次改配置后打快照

```bash
bash scripts/git_snapshot.sh "stage1 tune max_model_len and prefix cache metrics"
```

## 给正式实验打 tag

```bash
git tag -a exp-stage1-vllm-lite-pilot-001 -m "stage1 vllm qwen3 swebench lite pilot"
```

## 回滚配置/脚本

```bash
git log --oneline
git checkout <commit> -- configs scripts src README.md docs
```

## 保存实验结果

实验结果默认不进 git。需要长期归档时建议：

```bash
tar -czf runs_<run_id>.tar.gz runs/<run_id>
sha256sum runs_<run_id>.tar.gz > runs_<run_id>.tar.gz.sha256
```

如果要把 summary 进 git，只提交 `runs/<run_id>/analysis/summary.md` 的复制件到 `records/archive/`，不要提交全量 raw metrics 和 Docker 日志。
