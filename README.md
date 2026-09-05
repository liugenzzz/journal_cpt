# journal_cpt

期刊多模态知识注入与指令增量预训练数据生成流水线。代码结构参考 `book_cpt`，但任务、页面结构、文章切分、双栏读序和图表上下文绑定逻辑按 `journal_prompt_spec.md` 重新实现。

## 支持任务

- `article_metadata_extraction`
- `two_column_reading_order_reconstruction`
- `page_to_journal_layout_description`
- `section_heading_scope_alignment`
- `section_keypoint_summary`
- `figure_table_formula_to_text`
- `method_experiment_condition_extraction`
- `evidence_to_claim_chain`
- `article_contribution_conclusion`
- `cross_page_article_context`
- `domain_knowledge_corpus`

默认生成全部任务。可通过 `--task` 或 `--tasks` 只生成指定任务，并同步收窄路由、VLM provider 白名单、中间样本和导出文件范围。

## 运行

### Windows

```powershell
py -m journal_cpt.app.cli --input-dir journal_cpt\data --output-root journal_cpt\outputs --reuse-mineru
py -m journal_cpt.app.cli --journal "journal_cpt\data\2602.pdf" --output-root journal_cpt\outputs --reuse-mineru
py -m journal_cpt.app.cli --journal "journal_cpt\data\自由阻尼梁高频能量流响应的解析模型.pdf" --output-root journal_cpt\outputs --force-rebuild
py -m journal_cpt.app.cli --input-dir journal_cpt\data --output-root journal_cpt\outputs --tasks article_metadata_extraction,figure_table_formula_to_text
py -m journal_cpt.app.cli --input-dir journal_cpt\data --output-root journal_cpt\outputs --task domain_knowledge_corpus
```

### Linux

```bash
python3 -m journal_cpt.app.cli --input-dir journal_cpt/data --output-root journal_cpt/outputs --reuse-mineru
python3 -m journal_cpt.app.cli --journal "journal_cpt/data/2602.pdf" --output-root journal_cpt/outputs --reuse-mineru
python3 -m journal_cpt.app.cli --journal "journal_cpt/data/自由阻尼梁高频能量流响应的解析模型.pdf" --output-root journal_cpt/outputs --force-rebuild
python3 -m journal_cpt.app.cli --input-dir journal_cpt/data --output-root journal_cpt/outputs --tasks article_metadata_extraction,figure_table_formula_to_text
python3 -m journal_cpt.app.cli --input-dir journal_cpt/data --output-root journal_cpt/outputs --task domain_knowledge_corpus
```

流水线会在 MinerU/OCR 和页面渲染前自动尝试清洗 PDF 水印；命中水印时会生成 `preprocessed/{journal_id}_cleaned.pdf`，并强制重建相关缓存。若只想批量生成清洗后的 PDF，可运行：

```powershell
py clean_watermarked_pdfs.py outputs\watermark_cleaned data\自由阻尼梁高频能量流响应的解析模型.pdf data\等离子体合成射流激励器及其流动控制技术研究进展.pdf
```

```bash
python3 clean_watermarked_pdfs.py outputs/watermark_cleaned data/自由阻尼梁高频能量流响应的解析模型.pdf data/等离子体合成射流激励器及其流动控制技术研究进展.pdf
```

并发、MinerU 和 VLM provider pool 与 `book_cpt` 保持同一形态：

- `--journal-workers`：期刊 PDF 级并发数。
- `--page-workers`：页面渲染并发数。
- `--crop-workers`：块裁剪并发数。
- `--max-workers`：样本生成/大模型调用并发数。
- `--mineru-workers`、`--mineru-retry-count`、`--mineru-min-page-coverage`：MinerU 并发、重试和完整性阈值。
- `--reuse-mineru`、`--force-rebuild`、`--no-reuse-normalized`、`--no-reuse-samples`、`--no-reuse-exports`：断点续跑与缓存控制。

VLM provider 的 URL、model、并发和 fallback 策略与书籍侧保持一致。`journal_cpt/config.py` 会优先从同工作区的 `book_cpt.config` 继承匹配 provider 的 `api_key`；如果书籍配置不可用，则使用环境变量或无鉴权配置。

## 图表上下文绑定

`figure_table_formula_to_text` 不使用整页随意邻近文本作为图文上下文。路由时会为每个目标图、表或公式绑定：

- 目标视觉块自身。
- 同页最近且水平重叠的图注/表注。
- 与图号、表号或公式号相同的正文引用块。

生成提示词和本地校验都会保留 `context_binding_rule` 与 `figure_context_binding`，若目标块 ID 与绑定关系不一致，样本会被过滤。

## 输出

每个 PDF 对应一个输出目录：

- `manifest.jsonl`
- `pages/page_index.jsonl`
- `articles/articles.jsonl`
- `preprocessed/{journal_id}_cleaned.pdf`
- `preprocessed/watermark_cleaning.json`
- `images/pages/`
- `images/blocks/`
- `images/extracted/`
- `mineru/raw/`、`mineru/parsed/`、`mineru/status/`
- `normalized/`
- `samples/raw/{task_type}.jsonl`
- `samples/validated/{task_type}.jsonl`
- `samples/deduped/{task_type}.jsonl`
- `exports/sharegpt/{task_type}.jsonl`
- `exports/alpaca/{task_type}.jsonl`
- `exports/pt/domain_knowledge_corpus.jsonl`

PT 导出严格每行只包含 `text` 字段，不包含 instruction、input、output、messages、images 或 metadata。
