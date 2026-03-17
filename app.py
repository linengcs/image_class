"""
FastAPI 主入口 — 智能照片分类 Agent
  POST /api/upload     — 上传图片，启动处理任务
  GET  /api/status/{id} — SSE 实时推送进度
  GET  /api/result/{id} — 获取最终分类结果
  GET  /api/image/{id}/{filename} — 获取原始图片
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent import ImageItem, TaskStage, TaskState, run_pipeline

app = FastAPI(title="智能照片分类 Agent")

# ---------------------------------------------------------------------------
# 静态文件 & 全局状态
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# 内存中存储任务（demo 级别，生产环境应使用 Redis/DB）
tasks: dict[str, TaskState] = {}
progress_queues: dict[str, list[asyncio.Queue]] = {}


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    api_key: str | None = Form(None)
):
    """接收多张图片，启动后台处理."""
    if not files:
        return JSONResponse({"error": "请至少上传一张图片"}, status_code=400)

    task_id = uuid.uuid4().hex[:12]

    # 读取所有图片
    images: list[ImageItem] = []
    for f in files:
        raw = await f.read()
        images.append(ImageItem(filename=f.filename or f"image_{len(images)}.jpg", raw_bytes=raw))

    # 创建任务状态
    state = TaskState(task_id=task_id, images=images, api_key=api_key)
    tasks[task_id] = state
    progress_queues[task_id] = []

    # 进度回调：推送到所有监听的 SSE 队列
    def on_progress(stage: str, text: str):
        for q in progress_queues.get(task_id, []):
            q.put_nowait({"stage": stage, "text": text})

    state._on_progress = on_progress

    # 后台执行 pipeline
    asyncio.create_task(_run_task(state))

    return {"task_id": task_id, "image_count": len(images)}


async def _run_task(state: TaskState):
    """后台运行 pipeline."""
    await run_pipeline(state)
    # 通知所有 SSE 结束
    for q in progress_queues.get(state.task_id, []):
        await q.put(None)


@app.get("/api/status/{task_id}")
async def task_status_sse(task_id: str):
    """SSE 实时推送处理进度."""
    if task_id not in tasks:
        return JSONResponse({"error": "任务不存在"}, status_code=404)

    queue: asyncio.Queue = asyncio.Queue()
    progress_queues.setdefault(task_id, []).append(queue)

    # 先推送当前状态
    state = tasks[task_id]
    await queue.put({"stage": state.stage.value, "text": state.progress_text})

    async def event_generator():
        try:
            while True:
                msg = await asyncio.wait_for(queue.get(), timeout=120)
                if msg is None:
                    # 发送最终结果
                    yield f"data: {{\"stage\": \"done\", \"text\": \"处理完成！\"}}\n\n"
                    break
                yield f"data: {{\"stage\": \"{msg['stage']}\", \"text\": \"{msg['text']}\"}}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {{\"stage\": \"timeout\", \"text\": \"连接超时\"}}\n\n"
        finally:
            progress_queues.get(task_id, []).remove(queue) if queue in progress_queues.get(task_id, []) else None

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """获取最终分类结果."""
    if task_id not in tasks:
        return JSONResponse({"error": "任务不存在"}, status_code=404)

    state = tasks[task_id]

    if state.stage == TaskStage.ERROR:
        return JSONResponse({"error": state.error}, status_code=500)

    if state.stage != TaskStage.DONE:
        return JSONResponse(
            {
                "status": state.stage.value,
                "progress": state.progress_text,
            },
            status_code=202,
        )

    return {
        "task_id": task_id,
        "theme": state.theme,
        "categories": state.categories,
        "filtered_out": state.filtered_out,
        "total_images": len(state.images),
        "relevant_count": sum(1 for img in state.images if img.relevant),
    }


@app.get("/api/image/{task_id}/{filename}")
async def get_image(task_id: str, filename: str):
    """返回上传的原始图片."""
    if task_id not in tasks:
        return JSONResponse({"error": "任务不存在"}, status_code=404)

    state = tasks[task_id]
    for img in state.images:
        if img.filename == filename:
            from io import BytesIO
            return StreamingResponse(
                BytesIO(img.raw_bytes),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"},
            )

    return JSONResponse({"error": "图片不存在"}, status_code=404)


@app.get("/api/thumbnail/{task_id}/{filename}")
async def get_thumbnail(task_id: str, filename: str):
    """返回图片的缩略图（Agent 压缩后的较小版本），用于提升前端网格渲染性能."""
    if task_id not in tasks:
        return JSONResponse({"error": "任务不存在"}, status_code=404)

    state = tasks[task_id]
    for img in state.images:
        if img.filename == filename:
            from io import BytesIO
            if img.b64_uri and img.b64_uri.startswith("data:image/jpeg;base64,"):
                import base64
                b64_data = img.b64_uri.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)
                return StreamingResponse(
                    BytesIO(img_bytes),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"},
                )
            else:
                # 降级返回原图
                return StreamingResponse(
                    BytesIO(img.raw_bytes),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"},
                )

    return JSONResponse({"error": "缩略图不存在"}, status_code=404)


# ---------------------------------------------------------------------------
# 挂载前端静态文件
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
