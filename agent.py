"""
Agent Workflow — 三阶段 Pipeline
  Stage 1: 相关性过滤（批量）
  Stage 2: 主题识别
  Stage 3: 多级分类
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ImageAgent")

from llm_client import call_vision_json, image_to_base64


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class TaskStage(str, Enum):
    PENDING = "pending"
    FILTERING = "filtering"
    THEME = "theme"
    CLASSIFYING = "classifying"
    DONE = "done"
    ERROR = "error"


@dataclass
class ImageItem:
    filename: str
    raw_bytes: bytes
    b64_uri: str = ""
    relevant: bool = True


@dataclass
class TaskState:
    task_id: str
    images: list[ImageItem] = field(default_factory=list)
    api_key: str | None = None
    stage: TaskStage = TaskStage.PENDING
    progress_text: str = ""
    theme: dict[str, str] = field(default_factory=dict)
    categories: list[dict] = field(default_factory=list)
    filtered_out: list[str] = field(default_factory=list)
    error: str = ""

    # 进度回调
    _on_progress: Callable[[str, str], None] | None = None

    def set_progress(self, stage: TaskStage, text: str):
        self.stage = stage
        self.progress_text = text
        logger.info(f"[Task {self.task_id}] Stage: {stage.value} - {text}")
        if self._on_progress:
            self._on_progress(stage.value, text)


# ---------------------------------------------------------------------------
# Stage 1: 相关性过滤
# ---------------------------------------------------------------------------

FILTER_PROMPT_TEMPLATE = """你是一个照片分类助手。以下是用户手机相册中的 {count} 张照片（编号 {ids}）。

请判断每张照片是否与「外出旅游/参观景点」活动相关。

**应保留的照片：**
- 建筑物、景点、地标
- 景区内的景色、雕塑、展品
- 景区内拍摄的人物照
- 景区相关的标识牌、地图、门票

**应排除的照片：**
- 手机截图、聊天记录
- 纯食物照片（餐厅菜品特写）
- 与景点完全无关的日常照片
- 模糊不清无法辨认的照片

请严格返回 JSON 格式（不要包含其他文字）：
{{"relevant": [编号列表], "irrelevant": [编号列表]}}
"""

BATCH_SIZE = 6  # 每批发送的图片数量


async def stage_filter(state: TaskState) -> None:
    """阶段 1：过滤无关照片."""
    state.set_progress(TaskStage.FILTERING, "正在分析照片相关性...")

    # 先编码所有图片
    for img in state.images:
        if not img.b64_uri:
            img.b64_uri = image_to_base64(img.raw_bytes)

    # 准备批处理任务
    tasks = []
    batch_infos = []

    all_images = state.images
    for batch_start in range(0, len(all_images), BATCH_SIZE):
        batch = all_images[batch_start : batch_start + BATCH_SIZE]
        ids = list(range(batch_start, batch_start + len(batch)))
        id_str = ", ".join(str(i) for i in ids)

        prompt = FILTER_PROMPT_TEMPLATE.format(count=len(batch), ids=id_str)
        images_b64 = [img.b64_uri for img in batch]
        
        batch_infos.append(ids)
        tasks.append(asyncio.to_thread(call_vision_json, images_b64, prompt, api_key=state.api_key))

    state.set_progress(
        TaskStage.FILTERING,
        f"正在并发过滤 {len(all_images)} 张照片（分 {len(tasks)} 批）...",
    )

    # 并发执行
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果
    for ids, result in zip(batch_infos, results):
        if isinstance(result, Exception):
            logger.error(f"过滤任务出错: {result}")
            continue
            
        irrelevant_ids = result.get("irrelevant", [])
        for local_idx, global_idx in enumerate(ids):
            if global_idx in irrelevant_ids:
                all_images[global_idx].relevant = False

    # 汇总
    state.filtered_out = [img.filename for img in all_images if not img.relevant]
    relevant_count = sum(1 for img in all_images if img.relevant)
    state.set_progress(
        TaskStage.FILTERING,
        f"过滤完成：{relevant_count} 张相关照片，{len(state.filtered_out)} 张已排除",
    )


# ---------------------------------------------------------------------------
# Stage 2: 主题识别
# ---------------------------------------------------------------------------

THEME_PROMPT = """你是一个照片分析助手。以下是用户今天外出活动期间拍摄的照片。

请根据这些照片判断用户今天的主要活动主题。

请严格返回 JSON 格式（不要包含其他文字）：
{{"name": "活动主题（如：故宫参观、长城之旅）", "description": "一句话描述这次活动", "icon": "一个合适的emoji"}}
"""


async def stage_theme(state: TaskState) -> None:
    """阶段 2：识别主题."""
    state.set_progress(TaskStage.THEME, "正在识别活动主题...")

    relevant_images = [img for img in state.images if img.relevant]
    # 为节省成本，最多发 12 张图识别主题
    sample = relevant_images[:12]
    images_b64 = [img.b64_uri for img in sample]

    result = await asyncio.to_thread(call_vision_json, images_b64, THEME_PROMPT, api_key=state.api_key)
    state.theme = result
    state.set_progress(TaskStage.THEME, f"主题识别完成：{result.get('name', '未知')}")


# ---------------------------------------------------------------------------
# Stage 3: 多级分类
# ---------------------------------------------------------------------------

CLASSIFY_SCHEMA_PROMPT = """你是一个照片分类助手。以下是用户在「{theme}」期间拍摄的 {count} 张照片（编号 {ids}）。

请根据照片内容，设计一个合理的多级分类体系，并将每张照片归入对应类别。

分类要求：
1. 第一级按主要地点/区域分类（如：太和门、神武门、御花园）
2. 第二级按拍摄角度/内容类型分类（如：全景、细节、介绍牌、内部）
3. 每张照片只归入一个最合适的类别
4. 类别名称应简洁明了

请严格返回 JSON 格式（不要包含其他文字）：
{{
  "categories": [
    {{
      "name": "一级类别名",
      "subcategories": [
        {{
          "name": "二级类别名",
          "image_ids": [照片编号列表]
        }}
      ]
    }}
  ]
}}
"""


async def stage_classify(state: TaskState) -> None:
    """阶段 3：多级分类."""
    state.set_progress(TaskStage.CLASSIFYING, "正在进行多级分类...")

    relevant_images = [img for img in state.images if img.relevant]
    if not relevant_images:
        state.categories = []
        return

    theme_name = state.theme.get("name", "外出活动")

    # 分批进行分类，每批 10 张
    CLASSIFY_BATCH = 10
    tasks = []
    batch_infos = []

    for batch_start in range(0, len(relevant_images), CLASSIFY_BATCH):
        batch = relevant_images[batch_start : batch_start + CLASSIFY_BATCH]
        ids = list(range(batch_start, batch_start + len(batch)))
        id_str = ", ".join(str(i) for i in ids)

        prompt = CLASSIFY_SCHEMA_PROMPT.format(
            theme=theme_name, count=len(batch), ids=id_str
        )
        images_b64 = [img.b64_uri for img in batch]
        
        batch_infos.append(ids)
        tasks.append(asyncio.to_thread(call_vision_json, images_b64, prompt, api_key=state.api_key))

    state.set_progress(
        TaskStage.CLASSIFYING,
        f"正在并发分类 {len(relevant_images)} 张照片（分 {len(tasks)} 批）..."
    )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 合并各批次的分类体系
    merged_categories: dict[str, dict[str, list[str]]] = {}
    
    for ids, result in zip(batch_infos, results):
        if isinstance(result, Exception):
            logger.error(f"分类任务出错: {result}")
            continue
            
        categories = result.get("categories", [])
        for cat in categories:
            cat_name = cat.get("name", "其他")
            if cat_name not in merged_categories:
                merged_categories[cat_name] = {}
                
            for sub in cat.get("subcategories", []):
                sub_name = sub.get("name", "杂项")
                img_ids = sub.get("image_ids", [])
                
                if sub_name not in merged_categories[cat_name]:
                    merged_categories[cat_name][sub_name] = []
                    
                for idx in img_ids:
                    # 注意：llm 返回的是当前 batch 的相对/绝对编号，提示词里给的是全局 ids
                    if idx in ids:
                        # find correctly
                        merged_categories[cat_name][sub_name].append(relevant_images[idx].filename)

    # 转换回列表格式
    final_categories = []
    for cat_name, subs in merged_categories.items():
        sub_list = [{"name": sub_name, "images": imgs} for sub_name, imgs in subs.items() if imgs]
        if sub_list:
            final_categories.append({"name": cat_name, "subcategories": sub_list})

    state.categories = final_categories
    total_classified = sum(
        len(sub["images"])
        for cat in final_categories
        for sub in cat.get("subcategories", [])
    )
    state.set_progress(
        TaskStage.CLASSIFYING,
        f"分类完成：{len(final_categories)} 个大类明细，{total_classified} 张照片已归类",
    )


# ---------------------------------------------------------------------------
# 主 Pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(state: TaskState) -> None:
    """运行完整的三阶段 pipeline."""
    task_id = state.task_id
    logger.info(f"=== [Task {task_id}] 开始处理，共 {len(state.images)} 张图片 ===")
    try:
        await stage_filter(state)
        await stage_theme(state)
        await stage_classify(state)
        state.set_progress(TaskStage.DONE, "处理完成！")
        
        # 记录最终结果汇总
        logger.info(f"=== [Task {task_id}] 处理成功 ===")
        logger.info(f"[Task {task_id}] 主题: {state.theme.get('name', 'N/A')}")
        logger.info(f"[Task {task_id}] 分类数: {len(state.categories)}")
        logger.info(f"[Task {task_id}] 无关排除数: {len(state.filtered_out)}")
    except Exception as e:
        logger.error(f"=== [Task {task_id}] 处理失败: {e} ===")
        logger.error(traceback.format_exc())
        state.set_progress(TaskStage.ERROR, f"处理出错：{str(e)}")
        state.error = str(e)
