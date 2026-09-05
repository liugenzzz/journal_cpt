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

## VLM 调度

provider 之间是**抢占式**分配，不是先排名再指派：线程拿到任务后向所有可用 provider 逐个非阻塞试抢槽位，抢到哪个用哪个；全忙就等在条件变量上，谁先跑完释放槽位谁就接下一个任务。快的 provider 天然分到更多任务，不需要靠 `weight` 手动调。

- 单个 provider 的在途请求数严格不超过它的 `max_concurrency`。
- 某个 provider 失败时槽位立即归还，任务转投下一个 provider（受 `fallback.max_attempts` 限制）。
- `--journal-workers > 1` 时 journal 级走多进程，`VlmPool` 含线程锁无法 pickle，只能每个子进程各建一份。因此 `max_concurrency` 会按进程数向下摊薄、`min_interval_seconds` 按进程数放大，保证 provider 实际承受的并发不超过配置声明值。
- `--journal-workers 1`（默认）时整批共用一个 pool。

provider 的**冷却状态跨进程共享**，落在 `output_root/.runtime/vlm_cooldown/<provider>.cooldown`。一个 provider 挂掉后所有 journal 进程都会跳过它，不用各自再踩一次。冷却是低频写、高频读，所以写立即落盘、读带 `runtime.cooldown_refresh_seconds`（默认 1 秒）的内存缓存；时间戳用 wall clock，因为 `time.monotonic()` 跨进程不可比。

## MinerU 实例池

`mineru.providers` 可以配多台 MinerU。槽位是 `output_root/.runtime/mineru_slots/<provider>/slot_N.lock` 文件锁，**跨进程乃至跨主机**（只要 `output_root` 是共享盘）都成立，因此并发上限对整批任务都有效，不像 VLM 侧的信号量只在进程内。

- 分配同样是抢占式：按空闲槽位比例排序，谁先空谁被拿走；起点按 pid 打散，避免多个进程每轮都从同一台开始抢。
- 单次解析失败会把该实例写进 `output_root/.runtime/mineru_cooldown/`，重试自动转到另一台。解析结果不完整（`MinerUParseIncompleteError`）不算实例故障——换台也一样，不会拉黑。
- `--mineru-workers` 会写进每个 provider 的 `max_concurrency`。
- 删掉 `providers` 则退回读 `mineru` 顶层的 `url`/`server_url`，即旧式单实例配置。

加第二台只需把 `config.py` 里 `mineru.providers` 的 `mineru_2` 填上地址、`enabled` 改成 `True`。

## 损坏 PDF 的处理

pypdf 打不开的 PDF（截断、加密、结构损坏）会被**跳过**，不计入失败：

- 水印清洗阶段抛 `UnreadablePdfError`，`_process_journal` 单独捕获，打一条 WARNING 后继续下一篇。
- 跳过的文件记进 `<output-root>/skipped_journals.jsonl`，含 `journal_id`、`source_pdf` 和具体原因。
- CLI 收尾会把「跳过」和「失败」分开统计，避免坏文件淹没真正需要排查的错误。

注意 `processing/ingest.py` 的 `page_count()` 读页数时有 pypdf → PyMuPDF → 正则三级兜底，所以坏文件照样能被扫描进来，拦截点在水印清洗这一步。PyMuPDF 能读而 pypdf 读不了的文件目前也一并跳过。

## Prompt payload 装配

任务提示词本身（`config.py` 的 `PROMPTS`、`tasks/generation.py` 的 `instruction_rule`）不做任何改动，只处理装配进 `Input JSON` 的数据部分：

- **逐字节去重**：同一份版面块会以 `template_input.layout_blocks`、`layout_blocks`、`source.page.blocks` 等多个键重复出现，整页正文也会以 `ocr_text` / `paragraph_text` / `page_ocr` / `page_paragraph_text` / `full_text` 反复出现。序列化后完全相同的子树只保留首次出现，其余替换成 `<同 xxx，内容不再重复>` 的短引用——键仍在、信息一条不少，只是不再重复。
- **字符预算**：`generation.max_prompt_chars`（默认 80000）。超了按体积从大到小省略 `context` 下的辅助字段并留下说明，`template_input` 永不裁剪。设为 0 关闭。

实测（24 块的双栏页）：

| 任务 | 改前 | 改后 |
|---|---|---|
| `page_to_journal_layout_description` | 123,943 | 40,117 |
| `figure_table_formula_to_text` | 94,835 | 42,124 |
| `cross_page_article_context` | 387,897 | 43,759 |
| `article_contribution_conclusion` | 824,903 | 56,337 |

四个任务全部靠去重就装进预算，没有触发省略，即没有丢任何内容。

`vlm_pool` 各 provider 的 `max_tokens` 是**输出**上限，不是上下文窗口。设成与 `max_model_len` 同量级会让服务端算出 `max_model_len - prompt_len` 为负而直接 400（`max_tokens must be at least 1, got -6196`），因此设为 8192。

## 模型返回 JSON 的容错

模型写多行 answer 时经常直接敲回车而不是 `\n`，标准 `json.loads` 会报 `Invalid control character` 让整条响应作废。解析统一走 `strict=False`，只放开「字符串内的未转义控制字符」，其余语法仍严格校验。解析彻底失败时错误信息会带上响应开头 200 字，方便判断模型到底返回了什么。

## 思维链（think）处理

除 MinerU 外的生成模型都是推理模型，默认会输出思维链，因此请求和响应两侧都做了处理：

- **请求侧**：`VlmClient` 会给每个 provider 的 `chat_template_kwargs` 补上 `enable_thinking: False`，provider 里显式写的字段优先。provider 配置成空 dict 或不写该字段也同样生效；确需保留思维链时在 provider 上设 `disable_thinking: False`。
- **响应侧**：只取 `message.content`（或流式的 `delta.content`），忽略 `reasoning_content` 和 list content 里 `type` 为 thinking/reasoning 的分片。
- **解析侧**：解析 JSON 前统一调用 `strip_reasoning()`，剥掉成对的 `<think>...</think>`、只有闭合标签的前缀式思维链，以及被 `max_tokens` 截断后只剩开标签的输出。这样思维链里出现 `[` 或 `{` 时，不会再让 `parse_json_array` / `parse_jsonl_objects` 从错误的位置开始解析。

如果服务端返回的内容全部落在 `reasoning_content` 里而 `content` 为空（通常是没关思考且被 `max_tokens` 截断），会抛出带排查提示的 `RuntimeError`，由 provider pool 走 fallback。

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
