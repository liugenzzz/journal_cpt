# 期刊多模态知识注入与指令增量预训练数据处理提示词规格

## 1. 总控提示词

你是一个多模态期刊数据处理专家、数据工程架构师和航空领域论文数据集构建专家。现在需要围绕 `期刊/data/` 目录中的期刊 PDF，设计并实现一套可扩展、可追溯、可断点续跑的数据处理流水线，将期刊中的页面图像、双栏版面结构、OCR/文本层、文章元数据、摘要、关键词、正文段落、章节、小节、图表、公式、实验/仿真条件、参考文献上下文和跨页承接关系，转化为可用于 Qwen3-VL-8B/32B-Instruct 领域知识注入增量预训练和指令增量预训练的数据。

期刊数据与书籍数据不同：

- 期刊正文通常为双栏排版，同一页的真实阅读顺序不是简单的从上到下。
- 文章首页常包含通栏题名、作者、单位、摘要、关键词、中图分类号、DOI、收稿日期等元信息，随后进入双栏正文。
- 期刊合刊 PDF 可能包含封面、编委会、目录、多篇文章和索引页；这些页面不应直接生成正文知识样本。
- 单篇论文 PDF 通常只有 5-8 页，章节粒度短，适合围绕“问题-方法-实验/仿真-结果-结论”建模，而不是书籍式章节知识体系。
- 扫描型 PDF 可能没有可抽取文本层，需要 OCR/版面解析器；在没有 OCR 时只生成低风险页图结构样本或跳过文本型任务。
- 部分航空期刊 PDF 带有跨正文、公式和图表区域的斜向水印；应在页面渲染、MinerU/OCR 和图表裁剪前优先清洗，避免水印被识别为正文或干扰视觉任务。
- 图、表、公式和图注/表注在期刊论文中承载核心证据，适合生成图表证据解读、实验条件抽取和证据到结论链条任务。

当前样例数据包括：

- `搭载BFE件的飞发集成研发试验研究_NormalPdf.pdf`：单篇论文，可抽文本，首页通栏元信息+双栏正文，正文页有跨栏表格。
- `安全性分析在民用飞机货舱烟雾探测系统设计.pdf`：单篇论文，可抽文本，系统安全性分析主题。
- `大型飞机荷兰滚模态特性及航向气动需求研究.pdf`：单篇论文，可抽文本，公式、图表和气动参数较多。
- `冲击对挖补修理复材试验件的影响研究_NormalPd.pdf`：单篇论文，可抽文本，实验方案、试件参数和结果图表较多。
- `冲偏出跑道影响等级确认方法研究_NormalPdf.pdf`：单篇论文，可抽文本，法规、风险等级和判据推理较多。
- `2026年第47卷第1期电子期刊.pdf`、`2026年第47卷第2期电子期刊.pdf`：整期期刊，包含封面、编委、目录和多篇文章。
- `2602.pdf`：扫描型/图像型 PDF，文本层极少或为空，需要 OCR 适配器。
- `自由阻尼梁高频能量流响应的解析模型.pdf`：单篇论文，存在 `hkxb.buaa.edu.cn` 斜向图片水印，且文本层乱码较多，应先清洗水印再送 OCR/版面解析。
- `等离子体合成射流激励器及其流动控制技术研究进展.pdf`：长篇综述，存在同类斜向图片水印和底部版权/网址噪声，应优先做对象级水印预处理。

系统整体处理流程：

`PDF -> 水印预清洗 -> 页面图片 -> 文本层/OCR/版面解析 -> 双栏阅读顺序恢复 -> 文章切分与元数据抽取 -> 中间结构标准化 -> 任务路由 -> 样本生成或 Qwen3-VL 生成 -> 样本校验 -> 去重 -> ShareGPT/Alpaca/PT 导出`

默认情况下，全部期刊任务都应参与生成。命令行入口必须支持自定义任务选择：未传入任务参数时保持全任务生成；传入 `--task` 或 `--tasks` 时，只生成指定任务，并同步收窄任务路由、模型调用白名单、中间样本文件和导出文件范围。

示例：

```powershell
python -m journal_processing.cli --input-dir 期刊/data --output-root 期刊/outputs --render-pages
python -m journal_processing.cli --input-dir 期刊/data --output-root 期刊/outputs --tasks article_metadata_extraction,figure_table_formula_to_text
python -m journal_processing.cli --input-dir 期刊/data --output-root 期刊/outputs --task domain_knowledge_corpus
```

## 2. 适合期刊的任务类型

书籍任务不能原样照搬到期刊。期刊更强调文章元数据、双栏读序、论文结构、图表证据、方法条件和结果结论链。推荐支持以下 11 类任务。

### 2.1 `article_metadata_extraction`：文章元数据抽取

基于论文首页或合刊中的文章首页，抽取题名、作者、机构、期刊名、年卷期、页码、DOI、摘要、关键词、中图分类号、文献标识码、收稿日期、基金项目等。该任务适合期刊，不适合普通书籍。

输出重点：

- 文章题名、作者、单位。
- 摘要与关键词。
- DOI、卷期、页码、栏目。
- 页面证据和低置信字段标记。

### 2.2 `two_column_reading_order_reconstruction`：双栏阅读顺序恢复

基于整页图片、文本块坐标和 OCR 文本，恢复真实阅读顺序。需要区分页眉页脚、通栏题名/摘要/表格、左栏正文、右栏正文、跨栏图表、图注和表注。该任务是期刊双栏数据的关键任务。

输出重点：

- 页面是否为双栏。
- 通栏块、左栏块、右栏块和跨栏图表的顺序。
- 被过滤的页眉页脚/页码。
- 不确定读序说明。

### 2.3 `page_to_journal_layout_description`：期刊页面结构化描述

基于整页图片、OCR 和版面结构，描述期刊页面类型、版面区域、文章角色、栏目、双栏结构、图表公式和主要内容。与书籍页图描述不同，本任务必须显式描述双栏、通栏区域和论文页角色。

页面类型至少包括：

- `cover`
- `editorial_board`
- `table_of_contents`
- `article_first_page`
- `article_body`
- `references`
- `advertisement_or_notice`
- `blank`
- `scan_only`
- `unknown`

### 2.4 `section_heading_scope_alignment`：论文小节标题-正文范围对齐

基于论文中的一级/二级小节标题和其后正文，判断标题管辖的正文范围、研究步骤、条件和结论是否一致。书籍中常见的章节对齐任务在期刊中应收窄为“小节级”，避免把整篇论文当成章节。

输出重点：

- 小节标题。
- 标题控制的正文块范围。
- 正文是否围绕该标题展开。
- 是否存在跨栏、跨页或图表插入导致的范围不确定。

### 2.5 `section_keypoint_summary`：论文小节要点摘要

基于一个完整小节或连续正文片段，生成忠实摘要、核心术语、方法/条件/结果要点。不同于书籍段落摘要，本任务更关注论文小节中的研究目的、方法步骤、参数条件、数据结果和结论。

### 2.6 `figure_table_formula_to_text`：图表/公式到证据化文字解读

基于图、表、公式、图注、表注及正文引用，生成证据化文字解读。期刊中的图表通常是结论支撑，必须区分：

- 图表可见信息。
- 图注/表注信息。
- 正文解释信息。
- 模型不能凭图表外知识补全的部分。

### 2.7 `method_experiment_condition_extraction`：方法、实验/仿真条件抽取

基于“方法、模型、试验、仿真、工况、计算条件、参数设置”等小节，抽取研究对象、变量、参数、边界条件、工况、评价指标和实验/仿真流程。该任务非常适合工程期刊。

### 2.8 `evidence_to_claim_chain`：论文证据到论点链条

基于摘要、正文、图表、公式、结果和结论，生成“证据片段 -> 中间解释 -> 论文论点/结论”的可追溯链条。期刊论文的训练重点不是泛泛总结，而是让模型学习如何从实验/仿真/计算证据支持结论。

### 2.9 `article_contribution_conclusion`：文章贡献与结论提炼

基于整篇论文或论文片段，提炼研究问题、方法贡献、关键发现、适用条件、局限性和结论证据。该任务替代书籍中的 `chapter_key_conclusions`。

### 2.10 `cross_page_article_context`：跨页论文上下文衔接

基于连续 2-4 页同一篇文章，归纳跨页延续的概念、表格、图示、公式推导、方法步骤或结论承接。必须依赖至少两页证据，不能把单页内容伪装成跨页样本。

### 2.11 `domain_knowledge_corpus`：领域知识预训练语料

基于恢复阅读顺序后的论文正文、摘要、图注、表注、公式和结论，生成适合 LLaMA-Factory/Qwen 系列 `stage: pt` 的纯文本 JSONL。最终导出每行只能包含 `text` 字段，不包含问答、图片、metadata 或 conversation 字段。

## 3. 不建议从书籍直接迁移的任务

- `chapter_key_conclusions`：期刊没有书籍式章节体系，应替换为 `article_contribution_conclusion`。
- 粗粒度 `title_body_alignment`：期刊中题名覆盖整篇文章，直接做题名-全文对齐价值低，应改为小节级 `section_heading_scope_alignment`。
- 纯单页 `page_content_restatement`：期刊页面常被双栏和图表切割，单页重述可能破坏文章逻辑，应改为小节摘要、跨页上下文或读序恢复。
- 过多封闭问答：不应把论文全部转成问答；领域注入更需要保留论文原始知识表达、论证结构和证据链。
- 参考文献列表生成任务：参考文献页通常是低价值索引文本，应默认过滤；正文中的引用上下文可作为证据保留。

## 4. 目标数据格式

### 4.1 统一中间样本 JSONL

所有任务在导出 ShareGPT/Alpaca/PT 之前，应先保存为统一中间样本：

```json
{
  "id": "civil_aircraft_design_research_2025_04_p001_article_metadata_extraction_000001",
  "task_type": "article_metadata_extraction",
  "input_payload": {
    "source_text": "……",
    "page_images": ["期刊/outputs/.../images/pages/p001.png"],
    "layout_blocks": []
  },
  "output_payload": {
    "title": "搭载BFE件的飞发集成研发试验研究",
    "authors": ["王晶", "孙家琛", "高锋", "陈彬"],
    "abstract": "……",
    "keywords": ["飞发集成", "研发试验", "航空发动机", "BFE件", "功率提取"]
  },
  "evidence": {
    "source_pages": [1],
    "evidence_block_ids": ["p001_b007", "p001_b010", "p001_b011"],
    "evidence_text": ["……"],
    "visual_evidence": []
  },
  "metadata": {
    "source_pdf": "期刊/data/搭载BFE件的飞发集成研发试验研究_NormalPdf.pdf",
    "journal_id": "搭载bfe件的飞发集成研发试验研究_normalpdf",
    "article_id": "搭载bfe件的飞发集成研发试验研究",
    "article_title": "搭载BFE件的飞发集成研发试验研究",
    "journal_name": "民用飞机设计与研究",
    "year": "2025",
    "issue": "4",
    "page_index": 1,
    "page_type": "article_first_page",
    "generator": "heuristic_or_qwen3_vl",
    "quality_score": 0.86,
    "pipeline_version": "journal_prompt_v1"
  }
}
```

### 4.2 ShareGPT 格式

多模态任务应包含 `images` 和 `<image>` 占位符。纯文本任务可不包含图片。

```json
{
  "id": "journal_p002_two_column_reading_order_reconstruction_000001",
  "images": ["期刊/outputs/.../images/pages/p002.png"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n请根据该期刊页面图片、OCR 文本块和坐标，恢复双栏阅读顺序。"
    },
    {
      "from": "gpt",
      "value": "{\"column_mode\":\"two_column\",\"reading_order\":[...]}"
    }
  ],
  "metadata": {
    "task_type": "two_column_reading_order_reconstruction",
    "page_type": "article_body"
  }
}
```

### 4.3 Alpaca 格式

```json
{
  "instruction": "请从以下期刊论文首页内容中抽取文章元数据。",
  "input": "页面 OCR 与版面块：……",
  "output": "{\"title\":\"……\",\"authors\":[\"……\"]}",
  "images": ["期刊/outputs/.../images/pages/p001.png"],
  "metadata": {
    "task_type": "article_metadata_extraction"
  }
}
```

### 4.4 PT 纯文本格式

用于继续预训练的最终 JSONL 每行必须只包含 `text` 字段：

```json
{"text": "期刊：民用飞机设计与研究\n文章：搭载BFE件的飞发集成研发试验研究\n页码：140-141\n\n摘要：……\n关键词：……\n\n0 引言\n……"}
```

不得包含 `instruction`、`input`、`output`、`messages`、`conversations`、`images` 或 `metadata`。

## 5. 推荐目录与命名规范

```text
期刊/
  data/
    *.pdf
  outputs/
    manifest.jsonl
    {journal_id}_{pdf_name_safe}/
      preprocessed/
        {journal_id}_cleaned.pdf
        watermark_cleaning.json
      pages/
        page_index.jsonl
      images/
        pages/
          p001.png
          p002.png
        blocks/
          p001/
            p001_b007_text.png
      normalized/
        p001.json
        p002.json
      articles/
        articles.jsonl
      samples/
        raw/
          article_metadata_extraction.jsonl
          two_column_reading_order_reconstruction.jsonl
          page_to_journal_layout_description.jsonl
          section_heading_scope_alignment.jsonl
          section_keypoint_summary.jsonl
          figure_table_formula_to_text.jsonl
          method_experiment_condition_extraction.jsonl
          evidence_to_claim_chain.jsonl
          article_contribution_conclusion.jsonl
          cross_page_article_context.jsonl
          domain_knowledge_corpus.jsonl
        validated/
          {task_type}.jsonl
        deduped/
          {task_type}.jsonl
      exports/
        sharegpt/
          {task_type}.jsonl
        alpaca/
          {task_type}.jsonl
        pt/
          domain_knowledge_corpus.jsonl
      logs/
        pipeline.log
        errors.jsonl
        metrics.json
        checkpoints.json
```

样本 ID 建议格式：

```text
{journal_id}_{article_id}_p{page_no}_{block_or_window_id}_{task_type}_{sample_no}
```

如果无法识别文章，则使用：

```text
{journal_id}_article_unknown_p{page_no}_{block_or_window_id}_{task_type}_{sample_no}
```

## 5.1 水印预清洗

水印清洗应作为 PDF 进入页面渲染和 MinerU/OCR 前的预处理步骤，默认保留原始 PDF，只在输出目录生成清洗副本和清洗报告。对于 `hkxb.buaa.edu.cn` 这类航空期刊水印，优先使用 PDF 对象级处理：识别跨多数页面重复出现的 Form XObject 或图片 XObject，例如 `/Fm0` 中嵌套的固定尺寸水印图片，并从页面内容流中删除对应 `Do` 绘制调用。

推荐策略：

- 对象级删除作为主方案：命中重复 Form/图片水印时输出 `preprocessed/{journal_id}_cleaned.pdf`，后续页面图片、MinerU/OCR、块裁剪和视觉样本都使用清洗副本。
- OCR 后文本兜底过滤：若水印仍被识别为文本块，应按 `hkxb.buaa.edu.cn`、`航空学报编辑部`、`Acta Aeronautica et Astronautica Sinica` 等模式标记为 `watermark`，不进入 `visible_text`、PT 语料或训练样本证据。
- 缓存处理：一旦命中水印清洗，应强制重渲染页面图，并关闭旧 MinerU、normalized、sample 和 export 缓存复用，避免继续使用带水印的历史中间结果。
- 风险控制：不要把灰度阈值擦除作为默认主方案，因为它可能破坏浅灰网格线、曲线、公式细笔画和表格线；仅在扫描件或无法访问 PDF 对象结构时作为 fallback。
- 可追溯记录：清洗报告应记录 `source_pdf`、`cleaned_pdf`、页数、候选水印对象名、删除调用次数和是否清洗成功。

## 6. 标准化页面结构

每页应保存为 `JournalPageRecord`：

```json
{
  "journal_id": "civil_aircraft_design_research_2025_04",
  "source_pdf": "期刊/data/搭载BFE件的飞发集成研发试验研究_NormalPdf.pdf",
  "page_index": 1,
  "page_label": "140",
  "page_type": "article_first_page",
  "article_id": "bfe_fa_integrated_rd_test",
  "article_title": "搭载BFE件的飞发集成研发试验研究",
  "article_role": "first_page",
  "page_image": "期刊/outputs/.../images/pages/p001.png",
  "width": 612.3,
  "height": 824.9,
  "column_mode": "mixed_full_width_and_two_column",
  "blocks": [
    {
      "block_id": "p001_b007",
      "block_type": "text",
      "semantic_role": "article_title",
      "bbox": [107.2, 117.4, 505.1, 175.2],
      "text": "搭载BFE件的飞发集成研发试验研究",
      "reading_order": 7,
      "column": "full_width",
      "span_kind": "full_width",
      "confidence": 0.95,
      "image_path": null
    }
  ],
  "reading_order_blocks": ["p001_b007", "p001_b008", "p001_b010"],
  "full_text": "按双栏读序恢复后的正文……",
  "titles": [],
  "figures": [],
  "tables": [],
  "formulas": [],
  "semantic_links": [],
  "prev_page": null,
  "next_page": 2
}
```

版面块字段要求：

- `block_id`
- `block_type`: `text | title | section_title | figure | figure_caption | table | table_caption | formula | header | footer | page_number | reference | image | unknown`
- `semantic_role`: `journal_header | article_title | authors | affiliation | abstract | keywords | section_heading | body | figure_caption | table_caption | formula | reference | footer | page_number | toc_entry | unknown`
- `bbox`
- `text`
- `reading_order`
- `column`: `left | right | full_width | unknown`
- `span_kind`: `single_column | full_width | cross_column | floating`
- `confidence`
- `image_path`

双栏读序规则：

1. 页眉、页脚、页码默认进入结构记录，但不进入 PT 正文语料。
2. 通栏题名、作者、摘要、关键词、跨栏表格、跨栏图像按纵向位置插入。
3. 同一双栏区域内，按纵向带恢复：先读该带左栏从上到下，再读该带右栏从上到下。
4. 如果跨栏图表位于页面中部，跨栏图表是当前纵向带的分隔点；读左栏时一旦到达跨栏图表位置，应先转读同一纵向带内右栏内容，不要继续读跨栏图表下方的左栏内容；读完该带右栏后，再读图表/图注，再进入图表下方的新双栏区域。
5. 如果版面解析顺序与坐标顺序冲突，以坐标恢复的读序为准，并保留 `uncertainty_notes`。

## 7. 任务路由规则

- `article_metadata_extraction`：仅在文章首页或含摘要/关键词/题名的页面生成。
- `two_column_reading_order_reconstruction`：页面存在左右栏正文、跨栏块或读序风险时生成。
- `page_to_journal_layout_description`：页面图片存在，且不是空白页时生成。
- `section_heading_scope_alignment`：存在清晰小节标题及其后正文时生成。
- `section_keypoint_summary`：小节或页面正文长度足够，且不是目录/编委/参考文献页时生成。
- `figure_table_formula_to_text`：存在图注、表注、公式、表格或图表正文引用时生成。
- `method_experiment_condition_extraction`：页面/小节包含“方法、模型、试验、仿真、工况、计算、参数、边界条件、评价指标”等内容时生成。
- `evidence_to_claim_chain`：页面存在结果性、结论性或证据性句子，并能追溯到正文/图表/公式时生成。
- `article_contribution_conclusion`：同一篇文章至少有摘要和正文，或有结论/结果小节时生成。
- `cross_page_article_context`：同一篇文章连续 2-4 页存在主题延续、表格跨页、图文跨页或公式推导承接时生成。
- `domain_knowledge_corpus`：恢复读序后的正文、摘要、图注、表注和公式可读时生成；目录、编委、参考文献列表、空白页、广告页默认过滤。

## 8. 任务生成提示词

### 8.1 文章元数据抽取

```text
你是航空期刊论文元数据抽取专家。请根据期刊论文首页图片、OCR 文本和版面块，抽取文章元数据。

要求：
1. 只使用页面中出现的信息，不要补充外部知识。
2. 题名、作者、机构、摘要、关键词、DOI、卷期、页码等字段如果缺失，请置为空或标记 uncertain。
3. 摘要和关键词应尽量保持原文，不要改写。
4. 如果页面是目录、编委会、封面或广告页，应输出空数组或说明不适合抽取。

输出 JSON 数组，每条包含：
- title
- authors
- affiliations
- journal_name
- year
- volume
- issue
- pages
- doi
- abstract
- keywords
- classification_no
- document_code
- funding
- source_pages
- evidence_block_ids
- confidence
```

### 8.2 双栏阅读顺序恢复

```text
你是期刊双栏版面阅读顺序恢复专家。请根据页面图片、文本块坐标和 OCR 文本，恢复该页真实阅读顺序。

要求：
1. 区分页眉、页脚、页码、通栏区域、左栏、右栏和跨栏图表。
2. 双栏正文区域按纵向带恢复，带内先左栏从上到下，再右栏从上到下。
3. 跨栏图表或表格按其纵向位置作为分隔点插入读序；读取左栏时若到达跨栏图表，应先转读同一纵向带内右栏内容，不要继续读取跨栏图表下方的左栏内容。
4. 输出每个内容块的 block_id、角色、列位置和文本摘录。
5. 不确定时说明原因，不要强行拼接明显断裂的内容。

输出 JSON 数组，每条包含：
- column_mode
- filtered_noise_blocks
- reading_order
- reconstructed_text
- uncertainty_notes
- source_pages
- confidence
```

### 8.3 期刊页面结构化描述

```text
你是期刊页面结构描述专家。请根据页面图片、OCR 和版面结构，生成该页面的结构化描述。

要求：
1. 判断页面类型：封面、编委、目录、文章首页、文章正文、参考文献、扫描页等。
2. 描述通栏区域、双栏区域、图表公式和页面主旨。
3. 明确哪些内容属于训练正文，哪些应过滤。
4. 不要引入页面外知识。

输出 JSON 数组，每条包含：
- page_type
- article_role
- column_mode
- main_topic
- layout_regions
- content_blocks
- trainable_content
- filtered_content
- source_pages
- confidence
```

### 8.4 小节标题-正文范围对齐

```text
你是论文小节结构对齐专家。请根据期刊页面或跨页片段，判断小节标题与其控制正文范围是否匹配。

要求：
1. 对每个小节标题找到其后的正文范围。
2. 判断正文是否围绕标题展开。
3. 标注跨栏、图表插入或跨页造成的不确定性。
4. 不要把整篇论文题名当成小节标题。

输出 JSON 数组，每条包含：
- heading
- heading_level
- controlled_block_ids
- scope_summary
- alignment_judgement
- mismatch_risk
- source_pages
- confidence
```

### 8.5 小节要点摘要

```text
你是航空期刊论文小节摘要专家。请根据一个完整小节或连续正文片段，生成忠实、短小、证据充分的要点摘要。

要求：
1. 摘要必须来自输入正文。
2. 保留关键术语、变量、公式名称、工况和评价指标。
3. 优先覆盖研究目的、方法、参数条件、结果和结论。
4. 不要生成问答，不要引入外部知识。

输出 JSON 数组，每条包含：
- section_title
- summary
- key_terms
- method_or_condition_points
- result_or_claim_points
- source_pages
- evidence_text
- confidence
```

### 8.6 图表/公式证据化解读

```text
你是航空论文图表公式解读专家。请根据图、表、公式、图注/表注和正文引用，生成证据化文字解读。

要求：
1. 区分图表可见信息、图注/表注信息和正文解释信息。
2. 表格应说明字段、行列含义和关键对比。
3. 曲线图应说明变量、趋势、对比和坐标含义；看不清时不要猜。
4. 公式应保留变量名和适用上下文，不要自行推导新公式。
5. 每条解读必须能回溯到页面、图表编号或正文证据。

输出 JSON 数组，每条包含：
- object_type
- object_label
- visible_structure
- caption_information
- text_explanation
- evidence_to_claim
- source_pages
- evidence_block_ids
- confidence
```

### 8.7 方法、实验/仿真条件抽取

```text
你是航空工程论文方法与实验条件抽取专家。请从论文正文中抽取研究对象、方法流程、实验/仿真条件和评价指标。

要求：
1. 只抽取输入中明确出现的条件、参数、工况、设备、变量和指标。
2. 不要补充外部工程常识。
3. 对缺失或 OCR 不清晰的信息标记 uncertain。
4. 保留原始单位、符号和变量名。

输出 JSON 数组，每条包含：
- research_object
- method_steps
- experimental_or_simulation_conditions
- variables_and_parameters
- evaluation_metrics
- assumptions_or_constraints
- source_pages
- evidence_text
- confidence
```

### 8.8 论文证据到论点链条

```text
你是论文证据链构建专家。请根据论文片段，生成从证据到论点/结论的可追溯链条。

要求：
1. 证据必须来自输入中的正文、图表、公式、摘要或结论。
2. 链条包含证据片段、中间解释和被支持的论点。
3. 不要凭常识补全缺失环节。
4. 如果证据不足，应输出空数组或降低 confidence。

输出 JSON 数组，每条包含：
- claim
- evidence_chain
- chain_steps
- supporting_figures_tables_formulas
- reasoning_scope
- source_pages
- evidence_text
- confidence
```

### 8.9 文章贡献与结论提炼

```text
你是航空期刊论文贡献与结论提炼专家。请根据整篇论文或论文片段，提炼研究问题、方法贡献、关键发现、适用条件和结论。

要求：
1. 贡献和结论必须由输入正文或图表证据支持。
2. 不要把摘要简单拆句，要建立研究问题、方法和发现之间的关系。
3. 如果只给出部分页面，应明确 scope 是 article_fragment。
4. 保留限制条件和适用范围。

输出 JSON 数组，每条包含：
- research_problem
- contributions
- key_findings
- conclusions
- conditions_or_limitations
- supporting_evidence
- source_pages
- confidence
```

### 8.10 跨页论文上下文衔接

```text
你是跨页期刊论文上下文归纳专家。请根据连续多页同一篇论文内容，归纳跨页延续的知识表达。

要求：
1. 样本必须依赖至少两页证据。
2. 说明每页在论证中的角色，如提出方法、给出表格、解释结果、承接结论。
3. 优先处理图文跨页、表格跨页、公式推导跨页、实验步骤跨页。
4. 如果多页之间没有清晰关联，应输出空数组。

输出 JSON 数组，每条包含：
- context_topic
- page_roles
- cross_page_summary
- continuity_relations
- key_points_by_page
- source_pages
- evidence_text_by_page
- confidence
```

### 8.11 领域知识预训练语料

```text
你是领域期刊数据清洗与预训练语料构建助手。你的任务是把恢复双栏阅读顺序后的期刊论文文本处理成适合继续预训练的高质量文本语料。

重要规则：
1. 不要生成问答格式，不要生成 instruction/input/output。
2. 不要添加原文没有的知识、案例、结论或解释。
3. 尽量保留论文的知识密度、术语、摘要、关键词、小节结构、公式、图注、表注和结论。
4. 可以修正明显 OCR 错字、断行、空格、页眉页脚、页码和重复标题。
5. 删除封面、编委会、目录、参考文献列表、广告、空白页、水印和无训练价值页面。
6. 每个文本块开头保留期刊名、文章题名、页码范围或小节标题。
7. 表格结构清晰时转成 Markdown 表格；结构混乱时转成忠实自然语言描述。
8. 图注、表注可以保留；没有图片内容时不要凭空描述图片。
9. 按语义完整性切块，每块约 1200-3600 个中文字符，不在句子、公式或表格中间截断。
10. 输出必须是 JSONL，每行一个 JSON 对象，只包含 text 字段。

输出格式示例：
{"text": "期刊：民用飞机设计与研究\n文章：搭载BFE件的飞发集成研发试验研究\n页码：140\n\n摘要：……\n关键词：……\n\n0 引言\n……"}
```

## 9. 质量控制规则

- 双栏读序样本必须检查通栏块、左栏、右栏和跨栏图表的顺序。
- 目录、编委、封面、广告、空白页不得进入 PT 语料。
- 合刊 PDF 必须尽量切分文章；无法切分时保留 `article_unknown` 并降低质量分。
- 扫描型 PDF 如果没有 OCR，不生成文本摘要、证据链、PT 语料。
- 图表解读必须引用图号/表号/公式号或正文证据。
- 方法条件抽取不得补充页面中没有的参数。
- 证据链的每一步都应有 `evidence_text` 或 `evidence_block_ids`。
- 文章贡献结论任务必须说明范围是整篇还是片段。
- ShareGPT 多模态样本的 `<image>` 数量要与 `images` 字段一致。
- PT 导出文件每行只能包含 `text` 字段。
- 同一页面同一任务样本数量应受控，避免模板化重复。

## 10. 最终代码生成总提示词

```text
你是资深 Python 数据工程师和多模态期刊数据集构建专家。请根据本规格，为 期刊/data/ 目录中的航空期刊 PDF 实现一套期刊多模态知识注入与指令增量预训练数据生成流水线。

必须支持：
1. 扫描输入 PDF，建立 manifest。
2. 对带重复图片/Form 水印的 PDF 进行对象级预清洗，生成清洗副本和报告。
3. 渲染页面图片，可配置 DPI/缩放。
4. 抽取 PDF 文本层；对扫描型 PDF 保留 OCR 适配器接口。
5. 标准化页面块，识别页眉、页脚、目录、文章首页、正文、参考文献、水印等页面类型。
6. 恢复双栏阅读顺序，保留通栏块、左栏、右栏、跨栏图表和不确定说明。
7. 从单篇论文和整期期刊中切分文章，抽取文章元数据。
8. 支持 11 类期刊任务：article_metadata_extraction、two_column_reading_order_reconstruction、page_to_journal_layout_description、section_heading_scope_alignment、section_keypoint_summary、figure_table_formula_to_text、method_experiment_condition_extraction、evidence_to_claim_chain、article_contribution_conclusion、cross_page_article_context、domain_knowledge_corpus。
9. 默认生成全部任务；支持 --task 或 --tasks 指定子集。
10. 支持 Qwen3-VL-8B/32B-Instruct 或 OpenAI-compatible VLM provider；没有模型服务时可生成启发式样本和模型调用输入。
11. 输出统一中间样本、ShareGPT、Alpaca 和 PT JSONL。
12. 所有样本保留 source_pdf、journal_id、article_id、page_index、block_id、task_type、page_type、pipeline_version 等溯源字段。
13. 支持流式写入、断点续跑、失败页跳过、日志和基础指标。

请优先保证：
- 双栏读序正确。
- 目录/编委/参考文献等低价值页面被过滤。
- 斜向水印、版权网址和 OCR 误识别水印块不会进入正文、图文证据或 PT 语料。
- 图表、公式、实验条件和结论证据能被回溯。
- PT 语料只包含 text 字段。
- 代码结构清晰，模块可替换，方便后续接入 MinerU/OCR/Qwen3-VL。
```

## 11. 本地代码实现映射

本规格对应的期刊处理代码位于 `期刊/journal_processing/`，采用纯标准库优先的模块化结构，当前版本已实现可离线运行的启发式样本生成，并保留 PyMuPDF、MinerU/OCR、Qwen3-VL/OpenAI-compatible VLM 的替换入口。

代码模块对应关系：

- `journal_processing/processing/ingest.py`：PDF 扫描、页数估计、哈希、manifest 和 `journal_id` 生成。
- `journal_processing/processing/watermark.py`：识别重复 Form/Image 水印，生成清洗 PDF 和水印清洗报告。
- `journal_processing/processing/pdf_backends.py`：PDF 解析适配层，支持 `auto`、`pymupdf`、`probe_json`、`null` 后端；后续 MinerU/OCR 只需实现同样的 `RawPdfPage/RawPdfBlock` 输出。
- `journal_processing/processing/normalize.py`：期刊页面标准化、页面类型识别、语义块识别、双栏/通栏判定、阅读顺序恢复、文章切分和元数据初抽取。
- `journal_processing/tasks/routing.py`：根据页面类型、文章上下文、小节标题、图表公式线索、方法关键词和结论关键词路由 11 类任务。
- `journal_processing/tasks/generation.py`：离线启发式样本生成；生成内容只来自输入文本、块坐标和页面证据，不引入外部知识。后续可替换为 Qwen3-VL 生成器。
- `journal_processing/tasks/validation.py`：格式校验、证据页校验、多页任务校验、PT 纯文本长度校验和图片路径校验。
- `journal_processing/tasks/dedup.py`：基于任务类型、页面、证据块和输出内容的去重与每页任务数量控制。
- `journal_processing/tasks/exporters.py`：导出 ShareGPT、Alpaca 和 `stage: pt` JSONL；PT 导出严格只保留 `text` 字段。
- `journal_processing/app/pipeline.py`：端到端流水线；在渲染和 MinerU/OCR 前接入水印清洗，命中清洗时强制刷新相关缓存。
- `journal_processing/app/cli.py`：命令行入口。

推荐运行命令：

```powershell
python -m journal_processing.app.cli --input-dir E:\航天\期刊\data --output-root E:\航天\期刊\outputs
python -m journal_processing.app.cli --input-dir E:\航天\期刊\data --output-root E:\航天\期刊\outputs --task domain_knowledge_corpus
python -m journal_processing.app.cli --journal "搭载BFE件的飞发集成研发试验研究=E:\航天\期刊\data\搭载BFE件的飞发集成研发试验研究_NormalPdf.pdf" --output-root E:\航天\期刊\outputs_probe --parser probe_json --force-rebuild
```

当前机器若没有 PyMuPDF，可使用 `--parser probe_json` 基于 `_analysis/pdf_text_probe.json` 验证流程；生产处理建议安装 PyMuPDF 或接入 MinerU/OCR，以获得全量页面文本、坐标、页面图片和图表区域。
