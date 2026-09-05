from __future__ import annotations

import json


TASK_TYPES = [
    "article_metadata_extraction",
    "two_column_reading_order_reconstruction",
    "page_to_journal_layout_description",
    "section_heading_scope_alignment",
    "section_keypoint_summary",
    "figure_table_formula_to_text",
    "method_experiment_condition_extraction",
    "evidence_to_claim_chain",
    "article_contribution_conclusion",
    "cross_page_article_context",
    "domain_knowledge_corpus",
]

PROMPTS = {
    "system": (
        "你是多模态期刊数据处理专家、数据工程架构师和航空领域论文数据集构建专家。"
        "所有输出必须忠实基于输入证据，不引入外部知识；图文任务必须只使用与目标图、表或公式明确关联的上下文。"
        "处理图文任务时必须先观察当前图片中的可见内容，再与 OCR、版面块和绑定上下文交叉校验。"
        "解析正文时必须保持段落完整，遇到页末未结束并延续到下一页页首的段落，应按同一段落处理。"
    ),
    "article_metadata_extraction": (
        "你是航空期刊论文元数据抽取专家。请根据期刊论文首页图片、OCR 文本和版面块抽取文章元数据。"
        "只使用页面中出现的信息；缺失字段置空或标记 uncertain；目录、编委、封面或广告页返回 []。"
        "问题和答案必须由你根据当前期刊上下文生成，答案要用自然语言完整说明可见元数据，不要输出变量字段名。"
    ),
    "two_column_reading_order_reconstruction": (
        "你是期刊双栏版面阅读顺序恢复专家。请根据页面图片、文本块坐标和 OCR 文本恢复真实阅读顺序，"
        "区分页眉、页脚、页码、通栏区域、左栏、右栏和跨栏图表。"
        "论文首页如存在通栏摘要、关键词或元数据区域，必须先读取这些通栏内容；摘要下方进入双栏正文后，"
        "再按左栏从上到下、右栏从上到下恢复正文。"
        "双栏正文必须按纵向带恢复：读取左栏时若到达跨栏图表位置，必须先转读同一纵向带内的右栏内容，"
        "不得继续读取跨栏图表下方的左栏内容；读完该带右栏后，再读取跨栏图表，然后进入图表下方的新双栏带。"
        "生成的问题与答案必须忠于当前页面，答案要尽量覆盖该页所有可见正文、表格、图和公式。"
    ),
    "page_to_journal_layout_description": (
        "你是期刊页面结构描述专家。请判断页面类型、通栏区域、双栏区域、图表公式、页面主旨、"
        "可训练正文和应过滤内容。不要引入页面外知识。"
        "答案必须包含对页面可见内容的详细描述，尤其不要遗漏页面中出现的表、图、公式及其与正文的关系。"
    ),
    "section_heading_scope_alignment": (
        "你是论文小节结构对齐专家。请根据期刊页面或跨页片段判断小节标题与其控制正文范围是否匹配，"
        "标注跨栏、图表插入或跨页造成的不确定性。"
        "问题和答案必须忠于当前小节上下文，答案要完整说明标题控制范围、匹配依据和风险，不要输出变量字段名。"
    ),
    "section_keypoint_summary": (
        "你是航空期刊论文小节摘要专家。请根据完整小节或连续正文片段生成忠实摘要，"
        "优先覆盖研究目的、方法步骤、参数条件、数据结果和结论。"
        "问题和答案必须由你根据当前小节上下文生成，答案要尽量覆盖上下文全部关键信息，不要输出变量字段名。"
    ),
    "figure_table_formula_to_text": (
        "你是航空论文图表公式解读专家。请根据目标图、表、公式、图注/表注和正文引用生成证据化文字解读。"
        "必须区分可见信息、图注/表注信息和正文解释信息；只能使用与目标对象明确关联的上下文。"
        "必须利用大模型视觉能力对目标图片中的文字、结构、行列、曲线、箭头、符号或公式关系做详细描述，"
        "再结合绑定上下文生成忠实的问题与答案。"
        "生成 instruction 时不得出现图号、表号、公式号、图表题名、公式题名、论文题名或小节标题。"
    ),
    "method_experiment_condition_extraction": (
        "你是航空工程论文方法与实验条件抽取专家。请从论文正文中抽取研究对象、方法流程、实验/仿真条件、"
        "变量参数和评价指标。只抽取输入中明确出现的信息。"
        "问题和答案必须忠于当前论文片段，答案要自然说明对象、流程、条件、参数、指标和约束，不要输出变量字段名。"
    ),
    "evidence_to_claim_chain": (
        "你是论文证据链构建专家。请根据论文片段生成从证据到论点/结论的可追溯链条，"
        "每一步都必须有正文、图表、公式、摘要或结论证据支持。"
        "问题和答案必须忠于当前页面，答案要描述页面中可见的表、图、公式及其如何支持论点。"
    ),
    "article_contribution_conclusion": (
        "你是航空期刊论文贡献与结论提炼专家。请根据整篇论文或论文片段提炼研究问题、方法贡献、"
        "关键发现、适用条件和结论，并说明证据范围。"
        "问题和答案必须由你根据当前论文上下文生成，答案要覆盖研究问题、贡献、发现、结论、条件限制和证据范围，不要输出变量字段名。"
    ),
    "cross_page_article_context": (
        "你是跨页期刊论文上下文归纳专家。请根据连续多页同一篇论文内容归纳跨页延续的知识表达，"
        "必须依赖至少两页证据。"
        "问题和答案必须忠于连续页面图像与文本，答案要覆盖各页主要内容、图表/表格/公式和跨页承接关系。"
    ),
    "domain_knowledge_corpus": """
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
9. 按语义完整性切块，每块约 800-3600 个中文字符，不在句子、公式或表格中间截断。
10. 输出必须是 JSONL，每行一个 JSON 对象，只包含 text 字段。
""",
    "validator": "你是期刊论文训练样本质量校验专家。请检查样本是否符合 task_type、是否有证据支持、是否存在幻觉或格式错误。",
}

CFG = {
    "pipeline_version": "journal_prompt_v7_subfigure_groups",
    "encoding": "utf-8",
    "sharegpt_image_token": "<image>",
    "pdf_suffix": ".pdf",
    "pdf_page_regex": r"/Type\s*/Page\b",
    "output_dir_name": "outputs",
    "default_article_id": "article_unknown",
    "default_article_title": "",
    "default_block_id": "page",
    "default_block_type": "page",
    "default_semantic_role": "unknown",
    "unknown_journal_id": "untitled_journal",
    "timestamp_format": "%Y-%m-%dT%H:%M:%SZ",
    "hash_chunk_size": 1048576,
    "logger_name": "journal_cpt",
    "statuses": {"ready": "ready", "done": "done"},
    "paths": {
        "skipped_journals": "skipped_journals.jsonl",
        "manifest": "manifest.jsonl",
        "pages_manifest": "pages/page_index.jsonl",
        "articles_manifest": "articles/articles.jsonl",
        "page_images": "images/pages/p{page_no:03d}.png",
        "block_images": "images/blocks/p{page_no:03d}/{block_id}_{block_type}.png",
        "extracted_images": "images/extracted",
        "image_map": "mineru/image_map.json",
        "mineru_raw": "mineru/raw/{journal_id}.json",
        "mineru_parsed": "mineru/parsed/{journal_id}.json",
        "mineru_status": "mineru/status/{journal_id}.json",
        "normalized_page": "normalized/p{page_no:03d}.json",
        "sample_cache_state": "samples/cache_state.json",
        "sample_generation_state": "samples/generation_state.json",
        "sample_raw": "samples/raw/{task_type}.jsonl",
        "sample_validated": "samples/validated/{task_type}.jsonl",
        "sample_deduped": "samples/deduped/{task_type}.jsonl",
        "export_sharegpt": "exports/sharegpt/{task_type}.jsonl",
        "export_alpaca": "exports/alpaca/{task_type}.jsonl",
        "export_pt": "exports/pt/{task_type}.jsonl",
        "pipeline_log": "logs/pipeline.log",
        "errors": "logs/errors.jsonl",
        "metrics": "logs/metrics.json",
        "checkpoints": "logs/checkpoints.json",
    },
    "extracted_image_naming": {
        "template": "p{page_label}_img{image_no:03d}{suffix}",
        "unknown_page_label": "unknown",
        "allowed_suffixes": [".png", ".jpg", ".jpeg", ".webp", ".bmp"],
    },
    "task_types": TASK_TYPES,
    "export_formats": {
        "sharegpt": [
            "two_column_reading_order_reconstruction",
            "page_to_journal_layout_description",
            "figure_table_formula_to_text",
            "evidence_to_claim_chain",
            "cross_page_article_context",
        ],
        "alpaca": [
            "article_metadata_extraction",
            "section_heading_scope_alignment",
            "section_keypoint_summary",
            "method_experiment_condition_extraction",
            "article_contribution_conclusion",
        ],
        "pt": ["domain_knowledge_corpus"],
    },
    "block_types": {
        "title": "title",
        "section": "section_title",
        "section_title": "section_title",
        "text": "text",
        "paragraph": "text",
        "image": "figure",
        "figure": "figure",
        "figure_caption": "figure_caption",
        "caption": "figure_caption",
        "table": "table",
        "table_caption": "table_caption",
        "formula": "formula",
        "equation": "formula",
        "list": "list",
        "header": "header",
        "footer": "footer",
        "page_number": "page_number",
        "reference": "reference",
        "unknown": "unknown",
    },
    "semantic_roles": {
        "journal_header": "journal_header",
        "article_title": "article_title",
        "authors": "authors",
        "affiliation": "affiliation",
        "abstract": "abstract",
        "keywords": "keywords",
        "section_heading": "section_heading",
        "body": "body",
        "figure_caption": "figure_caption",
        "table_caption": "table_caption",
        "formula": "formula",
        "reference": "reference",
        "footer": "footer",
        "page_number": "page_number",
        "toc_entry": "toc_entry",
        "unknown": "unknown",
    },
    "page_types": {
        "trainable": ["article_first_page", "article_body"],
        "low_value": ["cover", "editorial_board", "table_of_contents", "references", "advertisement_or_notice", "blank"],
    },
    "runtime": {
        "input_dir": "journal_cpt/data",
        "input_journals": [],
        "output_root": "journal_cpt/outputs",
        "recursive": True,
        "journal_workers": 1,
        "max_workers": 4,
        "page_workers": 4,
        "crop_workers": 4,
        "vlm_max_pending": 8,
        "vlm_min_interval_seconds": 0.0,
        "cooldown_refresh_seconds": 1.0,
        "generation_state_flush_every": 20,
        "generation_state_flush_seconds": 10.0,
        "reuse_mineru": False,
        "reuse_normalized": True,
        "reuse_samples": True,
        "reuse_exports": True,
        "force_rebuild": False,
        "skip_vlm": False,
        "render_pages": True,
        "crop_blocks": False,
        "log_level": "INFO",
        "progress": True,
    },
    "render": {"dpi": 180, "image_format": "png", "max_side": 2200, "retry_count": 2},
    "crop_filter": {
        "enabled": True,
        "min_width": 28,
        "min_height": 32,
        "min_area": 3000,
        "min_text_chars": 10,
        "padding": 10,
        "text_block_types": ["title", "section_title", "text", "figure_caption", "table_caption", "list"],
        "visual_block_types": ["figure", "table", "formula"],
        "min_text_content_margin": 3,
        "edge_ink_band": 2,
        "max_edge_ink_ratio": 0.01,
        "min_visual_container_coverage": 0.75,
        "visual_container_overlap_ratio": 0.9,
        "white_pixel_threshold": 245,
        "max_blank_ratio": 0.985,
        "min_non_white_ratio": 0.005,
        "min_intensity_stddev": 3.0,
        "skip_block_types": ["header", "footer", "page_number", "unknown"],
        "skip_text_patterns": [r"^\s*\d+\s*$", r"^\s*第?\s*\d+\s*页\s*$"],
    },
    "mineru": {
        # 下面这些是所有 MinerU 实例共用的默认值；providers 里的条目只覆盖
        # url / server_url / max_concurrency 这类实例相关字段。
        "backend": "vlm-http-client",
        "parse_method": "auto",
        "lang_list": ["ch"],
        "timeout": 3600,
        "formula_enable": True,
        "table_enable": True,
        "return_md": True,
        "return_middle_json": True,
        "return_content_list": True,
        "return_images": True,
        "response_format_zip": False,
        "return_original_file": False,
        "max_concurrency": 16,
        "slot_poll_seconds": 2,
        "slot_stale_seconds": 7200,
        "retry_count": 3,
        "retry_backoff_seconds": 5,
        "retry_backoff_multiplier": 2,
        "cooldown_seconds": 300,
        "min_content_items": 1,
        "min_page_coverage": 0.75,
        "min_text_chars": 20,
        # MinerU 实例池。槽位是 output_root/.runtime/mineru_slots/<name>/ 下的文件锁，
        # 跨进程（乃至跨主机共享盘）都成立，谁先空谁被拿走。
        # 删掉 providers 会退回读 mineru 顶层的 url/server_url（旧式单实例配置）。
        "providers": [
            {
                "name": "mineru_1",
                "url": "http://10.107.226.27:8000",
                "server_url": "http://10.107.226.27:30000",
                "max_concurrency": 16,
                "weight": 1,
            },
            # 第二台 MinerU：把 url / server_url 换成实际地址，再把 enabled 改成 True。
            {
                "name": "mineru_2",
                "enabled": False,
                "url": "http://请填写第二台地址:8000",
                "server_url": "http://请填写第二台地址:30000",
                "max_concurrency": 16,
                "weight": 1,
            },
        ],
    },
    "vlm_pool": {
        "strategy": "least_busy_weighted_fallback",
        "fallback": {"enabled": True, "max_attempts": 2, "cooldown_seconds": 300},
        "providers": [
            {
                "name": "Qwen3.8-27B-1",
                "url": "http://jb-aionlineinferenceservice-161248564342717824-8000-nhss-job.z2120.nhss.zhejianglab.com:31080/v1/chat/completions",
                "model": "Qwen3.8-27B",
                "api_key_env": "",
                "api_key": "624pLvSLzgNqsmZo9wmQgn_JjRVaTpmJ73IZPE8QMUg",
                "stream": False,
                "temperature": 0.4,
                "max_tokens": 8192,
                "timeout": 2400,
                "chat_template_kwargs": {"enable_thinking": False},
                "capabilities": ["text", "image"],
                "task_types": TASK_TYPES,
                "weight": 1,
                "max_concurrency": 16,
            },
            {
                "name": "Qwen3.5-122B-A10B",
                "url": "http://jb-aionlineinferenceservice-161249674027102080-8000-nhss-job.z2120.nhss.zhejianglab.com:31080/v1/chat/completions",
                "model": "Qwen3.5-122B-A10B",
                "api_key_env": "",
                "api_key": "HjcAdBE1qv9BouiBCwu0SO0vgz2Rg7gCZPc2W--po5o",
                "stream": False,
                "temperature": 0.6,
                "max_tokens": 8192,
                "timeout": 2400,
                "capabilities": ["text", "image"],
                "task_types": TASK_TYPES,
                "weight": 2,
                "max_concurrency": 16,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            {
                "name": "Qwen3.8-2.4T-A95B-FP8",
                "url": "http://jb-aionlineinferenceservice-161253868802961280-8000-nhss-job.z2120.nhss.zhejianglab.com:31080/v1/chat/completions",
                "model": "Qwen3.8-2.4T-A95B-FP8",
                "api_key": "lA31fl6wllSbTun-LOtNF9FJcKOCtvBntaWW3wxjePA",
                "stream": False,
                "temperature": 0.6,
                "max_tokens": 8192,
                "timeout": 2400,
                "chat_template_kwargs": {"enable_thinking": False},
                "capabilities": ["text", "image"],
                "task_types": TASK_TYPES,
                "weight": 3,
                "max_concurrency": 16,
            },

            {
                "name": "Qwen3.8-27B-2",
                "url": "http://jb-aionlineinferenceservice-161666866205977152-8000-nhss-job.z2120.nhss.zhejianglab.com:31080/v1/chat/completions",
                "model": "Qwen3.8-27B",
                "api_key": "34Lvm96deF5PAV_u4UEPt2HuhBqpO71LJx-ZOAismVY",
                "stream": False,
                "temperature": 0.6,
                "max_tokens": 8192,
                "timeout": 2400,
                "chat_template_kwargs": {"enable_thinking": False},
                "capabilities": ["text", "image"],
                "task_types": TASK_TYPES,
                "weight": 3,
                "max_concurrency": 16,
        },

        ],
    },
    "routing": {
        "enabled_tasks": {task_type: True for task_type in TASK_TYPES},
        "min_page_text_chars": 120,
        "min_block_text_chars": 12,
        "min_section_text_chars": 220,
        "min_article_text_chars": 800,
        "min_domain_corpus_text_chars": 300,
        "cross_page_window": 3,
        "cross_page_min_images": 2,
        "cross_page_max_images": 3,
        "article_window": 8,
        "domain_corpus_window": 2,
        "domain_corpus_target_input_chars": 3600,
        "visual_block_types": ["figure", "table", "formula"],
        "text_block_types": ["text", "paragraph", "list"],
        "title_block_types": ["title", "section_title"],
        "caption_block_types": ["figure_caption", "table_caption"],
        "ocr_block_types": ["title", "section_title", "text", "figure_caption", "table_caption", "table", "formula", "list"],
        "low_value_page_types": ["cover", "editorial_board", "table_of_contents", "references", "advertisement_or_notice", "blank"],
        "method_keywords": ["方法", "模型", "试验", "实验", "仿真", "工况", "计算", "参数", "边界条件", "评价指标", "测量", "流程"],
        "claim_keywords": ["结果", "表明", "说明", "证明", "可见", "因此", "结论", "提高", "降低", "影响", "满足", "验证"],
        "conclusion_keywords": ["结论", "贡献", "发现", "结果表明", "研究表明", "提出", "建立", "验证"],
        "figure_ref_patterns": [r"图\s*[0-9一二三四五六七八九十]+", r"表\s*[0-9一二三四五六七八九十]+", r"式\s*[（(]?\s*[0-9]+", r"Fig\.?\s*[0-9]+", r"Table\s*[0-9]+"],
    },
    "generation": {
        "samples_per_job": {task_type: 1 for task_type in TASK_TYPES},
        "max_page_context_chars": 5000,
        "max_block_context_chars": 3600,
        "max_neighbor_context_chars": 5000,
        "max_article_context_chars": 10000,
        "max_pt_context_chars": 10000,
        "max_related_context_blocks": 8,
        # prompt 装配后的字符预算。服务端 max_model_len=131072 token，
        # 这里留足输出和图片 token 的余量；超了会按体积省略 context 下的辅助字段。
        "max_prompt_chars": 80000,
        "max_image_bytes": 2097152,
        "max_image_side": 1600,
        "image_jpeg_quality": 85,
        "heuristic_fallback": {
            "enabled": True,
            "on_vlm_error": False,
            "on_skip_vlm": False,
        },
    },
    "validation": {
        "min_question_chars": 4,
        "min_answer_chars": 8,
        "default_quality_score": 0.82,
        "low_quality_score": 0.35,
        "max_samples_per_page_task": 8,
        "dedup_similarity_threshold": 0.92,
        "low_value_page_types": ["cover", "editorial_board", "table_of_contents", "references", "advertisement_or_notice", "blank"],
        "required_output_fields": {
            "article_metadata_extraction": ["title"],
            "two_column_reading_order_reconstruction": ["page_visual_description", "column_mode", "reading_order"],
            "page_to_journal_layout_description": ["page_visual_description", "page_type", "column_mode", "layout_regions"],
            "section_heading_scope_alignment": ["heading", "alignment_judgement"],
            "section_keypoint_summary": ["summary", "key_terms"],
            "figure_table_formula_to_text": ["object_type", "visual_description", "visible_structure", "caption_information", "evidence_to_claim"],
            "method_experiment_condition_extraction": ["research_object", "method_steps"],
            "evidence_to_claim_chain": ["page_visual_description", "claim", "evidence_chain", "chain_steps"],
            "article_contribution_conclusion": ["research_problem", "contributions", "conclusions"],
            "cross_page_article_context": ["page_visual_descriptions", "context_topic", "cross_page_summary"],
            "domain_knowledge_corpus": ["text"],
        },
        "min_pt_text_chars": 80,
        "image_required_tasks": [
            "two_column_reading_order_reconstruction",
            "page_to_journal_layout_description",
            "figure_table_formula_to_text",
            "evidence_to_claim_chain",
            "cross_page_article_context",
        ],
        "multi_page_tasks": ["cross_page_article_context"],
        "figure_context_required_tasks": ["figure_table_formula_to_text"],
        "figure_instruction_forbidden_patterns": [
            r"图\s*[0-9一二三四五六七八九十]+",
            r"表\s*[0-9一二三四五六七八九十]+",
            r"式\s*[（(]?\s*[0-9]+",
            r"Fig\.?\s*[0-9]+",
            r"Figure\s*[0-9]+",
            r"Table\s*[0-9]+",
            r"标题",
            r"题名",
        ],
        "quality_dimensions": {
            "enabled": True,
            "min_text_format_score": 0.72,
            "min_qa_relevance_score": 0.70,
            "min_pt_source_coverage_score": 0.70,
            "min_pt_output_source_ratio": 0.65,
            "max_missing_pt_section_headings": 0,
            "min_visual_dependency_score": 0.60,
            "min_image_question_correspondence_score": 0.60,
            "max_symbol_ratio": 0.32,
            "max_repeated_char_run": 8,
            "min_answer_specific_terms": 2,
            "min_visual_support_overlap_terms": 1,
            "mojibake_patterns": [
                r"\ufffd",
                r"锟斤拷",
                r"ï¿½",
                r"Ã.",
                r"Â.",
                r"â€",
                r"[□�]{2,}",
            ],
            "visual_dependency_terms": [
                "图",
                "图片",
                "图像",
                "页面",
                "版面",
                "布局",
                "视觉",
                "表格",
                "公式",
                "曲线",
                "坐标",
                "双栏",
                "通栏",
                "阅读顺序",
                "跨页",
                "连续",
                "结构",
                "可见",
                "显示",
            ],
            "unusable_answer_patterns": [
                r"无法回答",
                r"无法判断",
                r"无法确定",
                r"不能确定",
                r"信息不足",
                r"数据不足",
                r"图片不可用",
                r"图像不可用",
                r"看不清",
                r"未提供足够",
                r"不适合该任务",
            ],
            "availability_only_patterns": [
                r"图片质量",
                r"图像质量",
                r"数据可用",
                r"任务可用",
                r"无法评估该任务",
                r"仅能说明.*可用",
            ],
        },
    },
    "prompt_style_rules": {},
    "prompts": PROMPTS,
}


def _inherit_book_provider_secrets() -> None:
    try:
        from book_cpt.config import CFG as BOOK_CFG  # type: ignore
    except Exception:
        return
    book_providers = BOOK_CFG.get("vlm_pool", {}).get("providers", [])
    if not isinstance(book_providers, list):
        return
    for provider in CFG["vlm_pool"]["providers"]:
        if not isinstance(provider, dict):
            continue
        for book_provider in book_providers:
            if not isinstance(book_provider, dict):
                continue
            same_name = provider.get("name") and provider.get("name") == book_provider.get("name")
            same_url_model = provider.get("url") == book_provider.get("url") and provider.get("model") == book_provider.get("model")
            if same_name or same_url_model:
                if book_provider.get("api_key") and not provider.get("api_key"):
                    provider["api_key"] = book_provider["api_key"]
                break


_inherit_book_provider_secrets()

CONFIG_JSON = json.dumps(CFG, ensure_ascii=False, indent=2)
