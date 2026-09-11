#!/usr/bin/env python3
"""诊断批量标注 vision_crop 标签映射问题。

单张图片::

    python scripts/diagnose_vision_map.py ^
        --image "D:/data/8.jpg" ^
        --checkpoint "D:/models/yolov8n.pt" ^
        --labels scripts/examples/labels.example.json ^
        --class-filter person

文件夹批量（模拟批量标注扫目录）::

    python scripts/diagnose_vision_map.py ^
        --folder "C:/Users/user/Desktop/test/face_detect_demo/data" ^
        --checkpoint "D:/models/yolov8n.pt" ^
        --labels scripts/examples/labels.example.json ^
        --class-filter person ^
        --base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" ^
        --api-key "sk-xxx" ^
        --legacy-full-pool ^
        --batch-quick

批量结果写入 ``diag_output/batch_summary.json``，每张图详情在 ``diag_output/<文件名>/``。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ROOT = ROOT.parent / "LR-Agent" / "vendor" / "inference"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image

from app.agent.annotation.annotation_scope import AnnotationScope
from app.agent.annotation.heuristic_map_service import heuristic_map_boxes
from app.agent.annotation.label_candidate_resolver import (
    LabelPoolResult,
    resolve_effective_label_candidates,
)
from app.agent.annotation.map_labels_service import (
    _coords_are_normalized,
    _resize_image_for_llm,
    _vision_map_box_with_crop,
    map_detection_boxes_to_labels_unified,
)
from app.agent.annotation.map_validation_service import find_label_names_in_text
from app.core.config import get_settings


def discover_images_in_folder(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"文件夹不存在: {folder}")
    files = [
        p.resolve()
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        raise SystemExit(f"文件夹内未找到图片: {folder}")
    return files


@dataclass
class ImageDiagReport:
    image: str
    path: str
    box_count: int = 0
    pool_source: str = ""
    pool_labels: list[str] = field(default_factory=list)
    prod_mappings: list[dict[str, Any]] = field(default_factory=list)
    duplicate_labels: list[str] = field(default_factory=list)
    legacy_diffs: list[dict[str, Any]] = field(default_factory=list)
    pool_single_diffs: list[dict[str, Any]] = field(default_factory=list)
    stability_unstable: list[dict[str, Any]] = field(default_factory=list)
    full_image_labels: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not (
            self.duplicate_labels
            or self.legacy_diffs
            or self.pool_single_diffs
            or self.stability_unstable
        )


def _log(verbose: bool, msg: str = "", *, end: str = "\n") -> None:
    if verbose:
        print(msg, end=end)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_inference_python(explicit: str = "") -> str:
    if explicit.strip():
        return explicit.strip()
    env = os.environ.get("LR_AGENT_INFERENCE_PYTHON", "").strip()
    if env:
        return env
    return sys.executable


def _yolo_request(
    image_path: Path,
    checkpoint: str,
    *,
    conf: float,
    iou: float,
) -> dict[str, Any]:
    return {
        "kind": "yolo_detect",
        "imagePath": str(image_path),
        "model": {
            "checkpointPath": checkpoint,
            "params": {
                "confThreshold": conf,
                "iouThreshold": iou,
            },
        },
    }


def _run_yolo_direct(inference_root: Path, request: dict[str, Any]) -> dict[str, Any]:
    inference_str = str(inference_root.resolve())
    if inference_str not in sys.path:
        sys.path.insert(0, inference_str)
    from runners.yolo import run_yolo_detect

    return run_yolo_detect(request)


def _run_yolo_subprocess(
    inference_root: Path,
    python: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    if not inference_root.is_dir():
        raise SystemExit(f"未找到推理目录: {inference_root}")
    payload = json.dumps({"cmd": "run", "request": request}, ensure_ascii=False)
    proc = subprocess.run(
        [python, "server.py"],
        input=payload + "\n",
        capture_output=True,
        text=True,
        cwd=str(inference_root),
        timeout=180,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "YOLO 子进程失败\n"
            f"python={python}\n"
            f"stderr={proc.stderr[:800]}\n"
            f"stdout={proc.stdout[:800]}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("YOLO 子进程无输出")
    resp = json.loads(lines[-1])
    if not resp.get("ok"):
        raise SystemExit(f"YOLO 推理错误: {resp.get('error')}")
    return resp.get("result") or {}


def _items_to_boxes(items: list[dict], *, class_filter: str = "") -> list[dict]:
    filt = class_filter.strip().lower()
    boxes: list[dict] = []
    for idx, item in enumerate(items):
        class_name = str(item.get("className") or item.get("class_name") or "")
        if filt and class_name.lower() != filt:
            continue
        geom = item.get("geometry") or {}
        if "x" in geom:
            x, y = float(geom["x"]), float(geom["y"])
            w, h = float(geom["width"]), float(geom["height"])
        elif "cx" in geom:
            cx, cy = float(geom["cx"]), float(geom["cy"])
            w, h = float(geom["width"]), float(geom["height"])
            x, y = cx - w / 2, cy - h / 2
        else:
            continue
        boxes.append(
            {
                "box_index": len(boxes),
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "class_name": class_name,
                "confidence": float(item.get("confidence") or 0),
            }
        )
    return boxes


def detect_boxes_yolo(
    image_path: Path,
    checkpoint: str,
    *,
    conf: float = 0.7,
    iou: float = 0.5,
    class_filter: str = "",
    inference_python: str = "",
    verbose: bool = True,
) -> list[dict]:
    """调用 vendor/inference YOLO，返回与批量标注一致的归一化框列表。"""
    ckpt = Path(checkpoint).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"YOLO 权重不存在: {ckpt}")

    request = _yolo_request(image_path, str(ckpt), conf=conf, iou=iou)
    python = resolve_inference_python(inference_python)

    _log(verbose, f"\n=== 阶段 0：YOLO 检测（全图） ===")
    _log(verbose, f"  checkpoint: {ckpt}")
    _log(verbose, f"  conf={conf} iou={iou} python={python}")

    try:
        result = _run_yolo_direct(INFERENCE_ROOT, request)
    except Exception as exc:
        _log(verbose, f"  当前解释器 YOLO 直接调用失败 ({exc})，改用子进程推理...")
        result = _run_yolo_subprocess(INFERENCE_ROOT, python, request)

    items = result.get("items") or []
    raw_count = len(items)
    boxes = _items_to_boxes(items, class_filter=class_filter)

    classes = Counter(str(it.get("className") or "") for it in items)
    _log(verbose, f"  原始检测: {raw_count} 框  类别: {dict(classes)}")
    if class_filter:
        _log(verbose, f"  过滤 class={class_filter!r} 后保留: {len(boxes)} 框")
    else:
        _log(verbose, f"  保留全部: {len(boxes)} 框")

    for b in boxes:
        _log(
            verbose,
            f"  box_index={b['box_index']} {b['class_name']} "
            f"conf={b['confidence']:.2f} "
            f"x={b['x']:.3f} y={b['y']:.3f} w={b['width']:.3f} h={b['height']:.3f}",
        )

    if not boxes:
        hint = "尝试降低 --conf 或去掉 --class-filter"
        raise SystemExit(f"未检测到可用框。{hint}")

    return boxes


def load_boxes(
    args: argparse.Namespace,
    image_path: Path,
    out_dir: Path,
    *,
    verbose: bool = True,
) -> list[dict]:
    if args.boxes.strip():
        boxes = load_json(Path(args.boxes))
        if not isinstance(boxes, list):
            raise SystemExit("--boxes 必须是 JSON 数组")
        for i, b in enumerate(boxes):
            if "box_index" not in b:
                b["box_index"] = i
        _log(verbose, "\n=== 使用手动 boxes 文件 ===")
        _log(verbose, f"  路径: {Path(args.boxes).resolve()}  数量: {len(boxes)}")
        return boxes

    if not args.checkpoint.strip():
        raise SystemExit(
            "未指定 --boxes 时需加 --checkpoint 以 YOLO 自动检测全图所有框\n"
            "示例: --checkpoint D:/models/yolov8n.pt [--class-filter person]"
        )

    boxes = detect_boxes_yolo(
        image_path,
        args.checkpoint,
        conf=args.conf,
        iou=args.iou,
        class_filter=args.class_filter,
        inference_python=args.inference_python,
        verbose=verbose,
    )
    out_path = out_dir / "detected_boxes.json"
    out_path.write_text(json.dumps(boxes, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(verbose, f"  已保存: {out_path}")
    return boxes


def print_assignment_warnings(mappings: list[dict], candidates: list[dict]) -> None:
    labeled = [
        label_name_by_id(candidates, str(m.get("label_id") or ""))
        for m in mappings
        if m.get("label_id")
    ]
    dupes = [name for name, cnt in Counter(labeled).items() if cnt > 1 and name != "(空)"]
    if dupes:
        print(f"  [WARN] 同图内重复分配标签: {dupes} → 逐框独立映射，无唯一约束")


def crop_pil(image_bytes: bytes, box: dict, *, normalized: bool) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as original:
        img = original.convert("RGB")
        iw, ih = img.size
        x, y, w, h = float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"])
        if normalized:
            x, y, w, h = x * iw, y * ih, w * iw, h * ih
        left = max(0, min(int(x), iw - 1))
        top = max(0, min(int(y), ih - 1))
        right = max(left + 1, min(int(x + w), iw))
        bottom = max(top + 1, min(int(y + h), ih))
        return img.crop((left, top, right, bottom))


def encode_jpeg(img: Image.Image, max_side: int, quality: int = 85) -> tuple[bytes, tuple[int, int]]:
    resized = _resize_image_for_llm(img, max_side) if max_side else img
    buf = io.BytesIO()
    rgb = resized.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), rgb.size


def bytes_to_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def label_name_by_id(candidates: list[dict], lid: str) -> str:
    for c in candidates:
        if str(c.get("id")) == lid:
            return str(c.get("name") or lid)
    return lid or "(空)"


def build_llm(base_url: str, api_key: str, model: str, temperature: float | None = None) -> ChatOpenAI:
    temp = temperature if temperature is not None else get_settings().annotation_llm_temperature
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        streaming=False,
        temperature=temp,
        timeout=120,
    )


async def resolve_label_pool(
    llm: ChatOpenAI,
    *,
    all_candidates: list[dict],
    boxes: list[dict],
    image_bytes: bytes,
    user_request: str = "标注图片中的球员",
    intent_summary: str = "批量标注球员",
    verbose: bool = True,
) -> LabelPoolResult:
    """与生产 map 入口相同的标签候选池解析（scope + 文本 + preflight）。"""
    pool = await resolve_effective_label_candidates(
        llm,
        all_candidates=all_candidates,
        scope=AnnotationScope(),
        user_request=user_request,
        intent_summary=intent_summary,
        box_count=len(boxes),
        image_bytes=image_bytes,
    )
    if verbose:
        print("\n=== 标签候选池（与生产 map 一致） ===")
        print(f"  source={pool.source}  有效候选数={len(pool.candidates)}")
        print(f"  候选: {[c.get('name') for c in pool.candidates]}")
        if pool.excluded_names:
            print(f"  已排除: {pool.excluded_names[:20]}")
        if pool.preflight_label_ids:
            print(f"  preflight_ids: {pool.preflight_label_ids}")
    return pool


def open_answer_matches_label(
    open_text: str,
    label_name: str,
    candidates: list[dict],
) -> bool:
    """开放问答与标签名是否语义一致（支持中/英子串与 reason 同款名称匹配）。"""
    if not label_name or label_name == "(空)":
        return "无法确定" in open_text or "unknown" in open_text.lower()
    if label_name.lower() in open_text.lower():
        return True
    mentioned = find_label_names_in_text(open_text, candidates)
    return any(m.lower() == label_name.lower() for m in mentioned)


def mapping_by_index(mappings: list[dict]) -> dict[int, dict]:
    return {int(m.get("box_index", 0)): m for m in mappings}


async def ask_open_ended(llm: ChatOpenAI, data_url: str, prompt: str) -> str:
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
    ]
    resp = await llm.ainvoke(messages)
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content).strip()


async def ask_closed_set_one_box(
    llm: ChatOpenAI,
    *,
    box_index: int,
    box: dict,
    crop_data_url: str,
    candidates: list[dict],
    user_request: str,
    intent_summary: str,
) -> dict[str, Any]:
    return await _vision_map_box_with_crop(
        llm,
        box_index=box_index,
        box=box,
        crop_data_url=crop_data_url,
        candidates=candidates,
        user_request=user_request,
        intent_summary=intent_summary,
    )


def phase_export_crops(
    image_bytes: bytes,
    boxes: list[dict],
    out_dir: Path,
) -> list[dict[str, Any]]:
    print("\n=== 阶段 1：导出裁剪图（检查分辨率/压缩） ===")
    normalized = _coords_are_normalized(boxes)
    print(f"  坐标格式: {'归一化 0~1' if normalized else '像素'}")
    meta: list[dict[str, Any]] = []

    for box in boxes:
        idx = int(box.get("box_index", 0))
        crop = crop_pil(image_bytes, box, normalized=normalized)
        raw_w, raw_h = crop.size

        crop.save(out_dir / f"box{idx}_raw_{raw_w}x{raw_h}.png")

        for tag, max_side, q in [
            ("prod_768", 768, 85),
            ("chat_1280", 1280, 85),
            ("high_1280", 1280, 95),
        ]:
            jpeg_bytes, (jw, jh) = encode_jpeg(crop, max_side, q)
            fname = f"box{idx}_{tag}_{jw}x{jh}_q{q}.jpg"
            (out_dir / fname).write_bytes(jpeg_bytes)
            meta.append(
                {
                    "box_index": idx,
                    "variant": tag,
                    "raw_size": [raw_w, raw_h],
                    "jpeg_size": [jw, jh],
                    "jpeg_kb": round(len(jpeg_bytes) / 1024, 1),
                    "file": fname,
                }
            )
            print(
                f"  box{idx} {tag}: 原裁剪 {raw_w}x{raw_h} → JPEG {jw}x{jh}, "
                f"{len(jpeg_bytes) // 1024}KB"
            )

    (out_dir / "crop_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def phase_heuristic(boxes: list[dict], candidates: list[dict]) -> None:
    print("\n=== 阶段 2：启发式映射（无 LLM） ===")
    payload = [
        {
            "box_index": b.get("box_index", i),
            "class_name": b.get("class_name", "person"),
            "confidence": b.get("confidence"),
        }
        for i, b in enumerate(boxes)
    ]
    mappings = heuristic_map_boxes(payload, candidates)
    for m in mappings:
        name = label_name_by_id(candidates, m.get("label_id", ""))
        print(f"  box_index={m['box_index']} → {name!r} ({m.get('reason')})")
    if all(not m.get("label_id") for m in mappings):
        print("  [OK] 启发式未匹配到球员名（符合预期，说明错标多半来自视觉路径）")
    else:
        print("  [WARN] 启发式已产生匹配，检查检测 class_name 或 OCR")


async def phase_prod_pipeline(
    llm: ChatOpenAI,
    image_path: Path,
    boxes: list[dict],
    candidates: list[dict],
    concurrency: int,
) -> list[dict]:
    print(f"\n=== 阶段 3：生产 pipeline（concurrency={concurrency}） ===")
    t0 = time.perf_counter()
    result = await map_detection_boxes_to_labels_unified(
        llm,
        user_request="标注图片中的球员",
        intent_summary="批量标注球员",
        label_candidates=candidates,
        boxes=boxes,
        use_vision=True,
        image_absolute_path=str(image_path),
        vision_map_concurrency=concurrency,
    )
    elapsed = time.perf_counter() - t0
    print(f"  method={result.get('method')} ok={result.get('ok')} 耗时={elapsed:.1f}s")
    if result.get("label_pool_source"):
        print(f"  label_pool_source={result.get('label_pool_source')}")
    for m in result.get("mappings") or []:
        name = label_name_by_id(candidates, m.get("label_id", ""))
        print(f"  box_index={m['box_index']} → {name!r}  reason={m.get('reason')!r}")
    print_assignment_warnings(result.get("mappings") or [], candidates)
    return result.get("mappings") or []


async def phase_open_vs_closed(
    llm: ChatOpenAI,
    image_bytes: bytes,
    boxes: list[dict],
    all_candidates: list[dict],
    pool: LabelPoolResult,
    prod_mappings: list[dict],
    *,
    show_legacy_full_pool: bool = False,
) -> None:
    print("\n=== 阶段 4：开放问答 vs 闭集（生产候选池，与阶段 3 同配置） ===")
    normalized = _coords_are_normalized(boxes)
    prod_by_index = mapping_by_index(prod_mappings)
    pool_names = "、".join(str(c.get("name") or "") for c in pool.candidates)

    for box in boxes:
        idx = int(box.get("box_index", 0))
        crop = crop_pil(image_bytes, box, normalized=normalized)

        prod_bytes, _ = encode_jpeg(crop, 768, 85)
        chat_bytes, _ = encode_jpeg(crop, 1280, 85)
        prod_url = bytes_to_data_url(prod_bytes)
        chat_url = bytes_to_data_url(chat_bytes)

        open_prompt_cn = (
            "请识别图中篮球运动员是谁，只回答中文姓名，不要解释。"
            "若无法确定请回答：无法确定"
        )
        open_prompt_en = (
            f"Who is the basketball player? Answer with ONLY one label name from: "
            f"{pool_names}. Say 'unknown' if unsure."
        )
        open_cn = await ask_open_ended(llm, prod_url, open_prompt_cn)
        open_en = await ask_open_ended(llm, prod_url, open_prompt_en)
        open_chat = await ask_open_ended(llm, chat_url, open_prompt_cn)

        closed = await ask_closed_set_one_box(
            llm,
            box_index=idx,
            box=box,
            crop_data_url=prod_url,
            candidates=pool.candidates,
            user_request="标注图片中的球员",
            intent_summary="批量标注球员",
        )
        closed_name = label_name_by_id(all_candidates, closed.get("label_id", ""))

        prod_m = prod_by_index.get(idx, {})
        prod_name = label_name_by_id(all_candidates, str(prod_m.get("label_id") or ""))

        print(f"\n  --- box_index={idx} ---")
        print(f"  阶段3生产映射:     {prod_name!r}")
        print(f"  开放(中文):         {open_cn[:120]}")
        print(f"  开放(英文标签名):   {open_en[:80]}")
        print(f"  闭集(生产候选池):   {closed_name!r}  reason={closed.get('reason')!r}")

        if show_legacy_full_pool and len(pool.candidates) < len(all_candidates):
            legacy = await ask_closed_set_one_box(
                llm,
                box_index=idx,
                box=box,
                crop_data_url=prod_url,
                candidates=all_candidates,
                user_request="标注图片中的球员",
                intent_summary="批量标注球员",
            )
            legacy_name = label_name_by_id(all_candidates, legacy.get("label_id", ""))
            print(f"  闭集(全量候选baseline): {legacy_name!r}  reason={legacy.get('reason')!r}")

        if closed_name != prod_name:
            print("  [WARN] 单框闭集(生产池) 与 阶段3 生产映射不一致")
        elif not open_answer_matches_label(open_en, closed_name, pool.candidates):
            print("  [WARN] 开放(英文) 与 闭集(生产池) 不一致")
        if open_cn.strip() != open_chat.strip():
            print("  [WARN] 768 vs 1280 开放(中文) 结果不同 → 可能是分辨率/压缩问题")


async def phase_full_image(
    llm: ChatOpenAI,
    image_bytes: bytes,
    candidates: list[dict],
) -> None:
    print("\n=== 阶段 5：整图开放问答（模拟聊天看图） ===")
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        scale = min(1.0, 1280 / max(w, h))
        if scale < 1.0:
            rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=85, optimize=True)
        full_url = bytes_to_data_url(buf.getvalue())

    names = "、".join(str(c["name"]) for c in candidates)
    prompt = (
        f"图中有哪些篮球运动员？请列出你认识的名字（可能在：{names}）。"
        "只列姓名，无法确认的不要猜。"
    )
    ans = await ask_open_ended(llm, full_url, prompt)
    print(f"  整图回答: {ans[:300]}")


async def phase_repeat_stability(
    llm: ChatOpenAI,
    image_bytes: bytes,
    boxes: list[dict],
    all_candidates: list[dict],
    pool: LabelPoolResult,
    repeats: int = 3,
) -> None:
    print(f"\n=== 阶段 6：稳定性 — 生产候选池闭集映射 ×{repeats} ===")
    normalized = _coords_are_normalized(boxes)
    if not boxes:
        return

    for box in boxes:
        idx = int(box.get("box_index", 0))
        crop = crop_pil(image_bytes, box, normalized=normalized)
        prod_url = bytes_to_data_url(encode_jpeg(crop, 768, 85)[0])

        results: list[str] = []
        for _i in range(repeats):
            r = await ask_closed_set_one_box(
                llm,
                box_index=idx,
                box=box,
                crop_data_url=prod_url,
                candidates=pool.candidates,
                user_request="标注图片中的球员",
                intent_summary="批量标注球员",
            )
            name = label_name_by_id(all_candidates, r.get("label_id", ""))
            results.append(name)

        print(f"  box_index={idx} (池={len(pool.candidates)}): {results}")
        if len(set(results)) > 1:
            print("    [WARN] 多次结果不一致")
        else:
            print("    [OK] 多次结果一致")


async def collect_legacy_diffs(
    llm: ChatOpenAI,
    image_bytes: bytes,
    boxes: list[dict],
    all_candidates: list[dict],
    prod_mappings: list[dict],
) -> list[dict[str, Any]]:
    normalized = _coords_are_normalized(boxes)
    prod_by_index = mapping_by_index(prod_mappings)
    diffs: list[dict[str, Any]] = []
    for box in boxes:
        idx = int(box.get("box_index", 0))
        crop = crop_pil(image_bytes, box, normalized=normalized)
        prod_url = bytes_to_data_url(encode_jpeg(crop, 768, 85)[0])
        legacy = await ask_closed_set_one_box(
            llm,
            box_index=idx,
            box=box,
            crop_data_url=prod_url,
            candidates=all_candidates,
            user_request="标注图片中的球员",
            intent_summary="批量标注球员",
        )
        prod_name = label_name_by_id(all_candidates, str(prod_by_index.get(idx, {}).get("label_id") or ""))
        legacy_name = label_name_by_id(all_candidates, legacy.get("label_id", ""))
        if legacy_name != prod_name:
            diffs.append(
                {
                    "box_index": idx,
                    "prod": prod_name,
                    "legacy_full_pool": legacy_name,
                }
            )
    return diffs


async def collect_pool_single_diffs(
    llm: ChatOpenAI,
    image_bytes: bytes,
    boxes: list[dict],
    all_candidates: list[dict],
    pool: LabelPoolResult,
    prod_mappings: list[dict],
) -> list[dict[str, Any]]:
    normalized = _coords_are_normalized(boxes)
    prod_by_index = mapping_by_index(prod_mappings)
    diffs: list[dict[str, Any]] = []
    for box in boxes:
        idx = int(box.get("box_index", 0))
        crop = crop_pil(image_bytes, box, normalized=normalized)
        prod_url = bytes_to_data_url(encode_jpeg(crop, 768, 85)[0])
        closed = await ask_closed_set_one_box(
            llm,
            box_index=idx,
            box=box,
            crop_data_url=prod_url,
            candidates=pool.candidates,
            user_request="标注图片中的球员",
            intent_summary="批量标注球员",
        )
        prod_name = label_name_by_id(all_candidates, str(prod_by_index.get(idx, {}).get("label_id") or ""))
        closed_name = label_name_by_id(all_candidates, closed.get("label_id", ""))
        if closed_name != prod_name:
            diffs.append(
                {
                    "box_index": idx,
                    "prod": prod_name,
                    "pool_single": closed_name,
                }
            )
    return diffs


async def collect_stability_issues(
    llm: ChatOpenAI,
    image_bytes: bytes,
    boxes: list[dict],
    all_candidates: list[dict],
    pool: LabelPoolResult,
    repeats: int,
) -> list[dict[str, Any]]:
    normalized = _coords_are_normalized(boxes)
    unstable: list[dict[str, Any]] = []
    for box in boxes:
        idx = int(box.get("box_index", 0))
        crop = crop_pil(image_bytes, box, normalized=normalized)
        prod_url = bytes_to_data_url(encode_jpeg(crop, 768, 85)[0])
        names: list[str] = []
        for _ in range(repeats):
            r = await ask_closed_set_one_box(
                llm,
                box_index=idx,
                box=box,
                crop_data_url=prod_url,
                candidates=pool.candidates,
                user_request="标注图片中的球员",
                intent_summary="批量标注球员",
            )
            names.append(label_name_by_id(all_candidates, r.get("label_id", "")))
        if len(set(names)) > 1:
            unstable.append({"box_index": idx, "results": names})
    return unstable


def duplicate_labels_from_mappings(
    mappings: list[dict],
    candidates: list[dict],
) -> list[str]:
    labeled = [
        label_name_by_id(candidates, str(m.get("label_id") or ""))
        for m in mappings
        if m.get("label_id")
    ]
    return [name for name, cnt in Counter(labeled).items() if cnt > 1 and name != "(空)"]


def mappings_to_rows(mappings: list[dict], candidates: list[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in sorted(mappings, key=lambda x: int(x.get("box_index", 0))):
        rows.append(
            {
                "box_index": int(m.get("box_index", 0)),
                "label": label_name_by_id(candidates, str(m.get("label_id") or "")),
                "label_id": str(m.get("label_id") or ""),
                "reason": str(m.get("reason") or "")[:160],
            }
        )
    return rows


async def diagnose_single_image(
    args: argparse.Namespace,
    image_path: Path,
    candidates: list[dict],
    out_dir: Path,
    llm: ChatOpenAI | None,
    *,
    verbose: bool = True,
    batch_mode: bool = False,
) -> ImageDiagReport:
    report = ImageDiagReport(image=image_path.name, path=str(image_path))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        boxes = load_boxes(args, image_path, out_dir, verbose=verbose)
        image_bytes = image_path.read_bytes()
        report.box_count = len(boxes)

        _log(verbose, f"\n图片: {image_path}")
        _log(verbose, f"待测框数: {len(boxes)}  标签: {[c.get('name') for c in candidates]}")

        if verbose:
            phase_export_crops(image_bytes, boxes, out_dir)
            phase_heuristic(boxes, candidates)

        if args.skip_llm:
            return report

        if llm is None:
            raise RuntimeError("LLM 未初始化")

        prod_result = await map_detection_boxes_to_labels_unified(
            llm,
            user_request="标注图片中的球员",
            intent_summary="批量标注球员",
            label_candidates=candidates,
            boxes=boxes,
            use_vision=True,
            image_absolute_path=str(image_path),
            vision_map_concurrency=3,
        )
        prod_mappings = prod_result.get("mappings") or []
        report.pool_source = str(prod_result.get("label_pool_source") or "")
        report.prod_mappings = mappings_to_rows(prod_mappings, candidates)
        report.duplicate_labels = duplicate_labels_from_mappings(prod_mappings, candidates)

        if verbose and not batch_mode:
            print("\n=== 阶段 3：生产 pipeline（concurrency=3） ===")
            print(f"  method={prod_result.get('method')} ok={prod_result.get('ok')}")
            if report.pool_source:
                print(f"  label_pool_source={report.pool_source}")
            for row in report.prod_mappings:
                print(f"  box_index={row['box_index']} → {row['label']!r}")
            print_assignment_warnings(prod_mappings, candidates)
            if args.compare_concurrency:
                await phase_prod_pipeline(llm, image_path, boxes, candidates, concurrency=1)

        pool = await resolve_label_pool(
            llm,
            all_candidates=candidates,
            boxes=boxes,
            image_bytes=image_bytes,
            verbose=verbose and not batch_mode,
        )
        report.pool_labels = [str(c.get("name") or "") for c in pool.candidates]
        if not report.pool_source:
            report.pool_source = pool.source

        if batch_mode:
            if args.legacy_full_pool and len(pool.candidates) < len(candidates):
                report.legacy_diffs = await collect_legacy_diffs(
                    llm, image_bytes, boxes, candidates, prod_mappings
                )
            report.pool_single_diffs = await collect_pool_single_diffs(
                llm, image_bytes, boxes, candidates, pool, prod_mappings
            )
            stability_repeats = 1 if args.batch_quick else args.repeats
            report.stability_unstable = await collect_stability_issues(
                llm, image_bytes, boxes, candidates, pool, stability_repeats
            )
            if not args.batch_quick:
                report.full_image_labels = await phase_full_image_return(
                    llm, image_bytes, candidates
                )
        elif not args.batch_quick:
            await phase_open_vs_closed(
                llm,
                image_bytes,
                boxes,
                candidates,
                pool,
                prod_mappings,
                show_legacy_full_pool=args.legacy_full_pool,
            )
            await phase_full_image(llm, image_bytes, candidates)
            await phase_repeat_stability(
                llm, image_bytes, boxes, candidates, pool, repeats=args.repeats
            )

        (out_dir / "report.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        report.error = str(exc)
        if verbose:
            print(f"  [ERROR] {exc}")
    return report


async def phase_full_image_return(
    llm: ChatOpenAI,
    image_bytes: bytes,
    candidates: list[dict],
) -> str:
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        scale = min(1.0, 1280 / max(w, h))
        if scale < 1.0:
            rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=85, optimize=True)
        full_url = bytes_to_data_url(buf.getvalue())
    names = "、".join(str(c["name"]) for c in candidates)
    prompt = (
        f"图中有哪些篮球运动员？请列出你认识的名字（可能在：{names}）。"
        "只列姓名，无法确认的不要猜。"
    )
    return await ask_open_ended(llm, full_url, prompt)


def print_batch_summary(reports: list[ImageDiagReport], summary_path: Path) -> None:
    print("\n" + "=" * 72)
    print("批量诊断汇总")
    print("=" * 72)
    ok_n = sum(1 for r in reports if r.ok)
    print(f"图片数: {len(reports)}  通过: {ok_n}  有问题: {len(reports) - ok_n}")
    print(f"汇总 JSON: {summary_path}")
    print()
    header = f"{'图片':<16} {'框':>3} {'池':<10} {'生产映射':<28} {'问题'}"
    print(header)
    print("-" * 72)
    for r in reports:
        if r.error:
            print(f"{r.image:<16} {'—':>3} {'—':<10} {'—':<28} ERROR: {r.error[:40]}")
            continue
        labels = ",".join(f"{m['label']}" for m in r.prod_mappings) or "(无)"
        issues: list[str] = []
        if r.duplicate_labels:
            issues.append(f"重复:{','.join(r.duplicate_labels)}")
        if r.legacy_diffs:
            issues.append(f"legacy差{len(r.legacy_diffs)}")
        if r.pool_single_diffs:
            issues.append(f"单框差{len(r.pool_single_diffs)}")
        if r.stability_unstable:
            issues.append(f"不稳定{len(r.stability_unstable)}")
        issue_str = "; ".join(issues) if issues else "OK"
        print(f"{r.image:<16} {r.box_count:>3} {r.pool_source:<10} {labels:<28} {issue_str}")

    problem_reports = [r for r in reports if not r.ok and not r.error]
    if problem_reports:
        print("\n--- 问题详情 ---")
        for r in problem_reports:
            print(f"\n[{r.image}] pool={r.pool_labels}")
            for row in r.prod_mappings:
                print(f"  box{row['box_index']}: {row['label']}")
            for d in r.legacy_diffs:
                print(f"  legacy box{d['box_index']}: prod={d['prod']!r} vs full={d['legacy_full_pool']!r}")
            for d in r.pool_single_diffs:
                print(f"  pool单框 box{d['box_index']}: prod={d['prod']!r} vs single={d['pool_single']!r}")
            for s in r.stability_unstable:
                print(f"  不稳定 box{s['box_index']}: {s['results']}")


async def run_folder_batch(
    args: argparse.Namespace,
    candidates: list[dict],
    llm: ChatOpenAI,
) -> None:
    folder = Path(args.folder).resolve()
    images = discover_images_in_folder(folder)
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"文件夹: {folder}")
    print(f"图片数: {len(images)}")
    print(f"输出根目录: {out_root}")

    reports: list[ImageDiagReport] = []
    for i, image_path in enumerate(images):
        print(f"\n{'=' * 72}")
        print(f"[{i + 1}/{len(images)}] {image_path.name}")
        print("=" * 72)
        sub_out = out_root / image_path.stem
        report = await diagnose_single_image(
            args,
            image_path,
            candidates,
            sub_out,
            llm,
            verbose=not args.batch_quiet,
            batch_mode=True,
        )
        reports.append(report)
        if report.error:
            print(f"  -> ERROR: {report.error[:80]}")
        elif report.ok:
            labels = ",".join(m["label"] for m in report.prod_mappings) or "(无)"
            print(f"  -> OK  boxes={report.box_count}  {labels}")
        else:
            labels = ",".join(m["label"] for m in report.prod_mappings) or "(无)"
            issues = []
            if report.duplicate_labels:
                issues.append("重复")
            if report.legacy_diffs:
                issues.append(f"legacy×{len(report.legacy_diffs)}")
            if report.pool_single_diffs:
                issues.append(f"单框×{len(report.pool_single_diffs)}")
            if report.stability_unstable:
                issues.append(f"不稳定×{len(report.stability_unstable)}")
            print(f"  -> ISSUE({','.join(issues)})  boxes={report.box_count}  {labels}")

    summary_path = out_root / "batch_summary.json"
    summary_path.write_text(
        json.dumps([asdict(r) for r in reports], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_batch_summary(reports, summary_path)


def print_summary(out_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("结果解读指南")
    print("=" * 60)
    print("1. 看 diag_output/box*_raw_*.png 是否框准、脸是否清晰")
    print("2. 若 raw 很小（如 <80px）→ 小框未放大，LLM 输入质量差")
    print("3. 阶段 4/6 使用与生产相同的「标签候选池」（preflight/文本过滤后）")
    print("4. 若 阶段3 对、闭集(全量baseline) 错 → 干扰标签是主因（加 --legacy-full-pool 查看）")
    print("5. 若 768 vs 1280 开放(中文) 不同 → 可能是分辨率/压缩问题")
    print("6. 若启发式也有匹配 → 检查 YOLO class_name")
    print("7. 若同图重复分配同一球员 → 逐框独立映射，需联合分配策略")
    print(f"\n所有裁剪图已保存: {out_dir.resolve()}")
    detected = out_dir / "detected_boxes.json"
    if detected.is_file():
        print(f"检测框 JSON: {detected.resolve()}")


async def main_async(args: argparse.Namespace) -> None:
    candidates = load_json(Path(args.labels))
    if not isinstance(candidates, list):
        raise SystemExit("--labels 必须是 JSON 数组")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_llm:
        llm: ChatOpenAI | None = None
    else:
        llm = build_llm(args.base_url, args.api_key, args.model, args.temperature)
        print(f"模型: {args.model}  temperature={llm.temperature}")

    if args.folder:
        if args.skip_llm:
            raise SystemExit("文件夹批量模式需要 LLM（去掉 --skip-llm）")
        await run_folder_batch(args, candidates, llm)
        return

    if not args.image:
        raise SystemExit("请指定 --image 或 --folder")

    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise SystemExit(f"图片不存在: {image_path}")

    if args.skip_llm:
        await diagnose_single_image(
            args, image_path, candidates, out_dir, None, verbose=True, batch_mode=False
        )
        print("\n(--skip-llm 已跳过 LLM 调用)")
        print_summary(out_dir)
        return

    await diagnose_single_image(
        args, image_path, candidates, out_dir, llm, verbose=True, batch_mode=False
    )
    print_summary(out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="诊断 vision_crop 标签映射（支持 YOLO 自动检测全图所有框）",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", default="", help="单张图片绝对路径")
    src.add_argument(
        "--folder",
        default="",
        help="图片文件夹（批量测试，结果写入 output-dir/<文件名>/ 与 batch_summary.json）",
    )
    p.add_argument(
        "--boxes",
        default="",
        help="手动检测框 JSON；省略则用 --checkpoint 自动 YOLO 检测",
    )
    p.add_argument(
        "--checkpoint",
        default="",
        help="YOLO 权重 .pt 路径（未指定 --boxes 时必填）",
    )
    p.add_argument("--conf", type=float, default=0.7, help="YOLO 置信度阈值")
    p.add_argument("--iou", type=float, default=0.5, help="YOLO IoU 阈值")
    p.add_argument(
        "--class-filter",
        default="",
        help="只保留此类检测框，如 person（空=保留全部）",
    )
    p.add_argument(
        "--inference-python",
        default="",
        help="推理 Python 路径（默认 LR_AGENT_INFERENCE_PYTHON 或当前解释器）",
    )
    p.add_argument("--labels", required=True, help="标签候选 JSON")
    p.add_argument("--output-dir", default="./diag_output")
    p.add_argument("--base-url", default="", help="如 DashScope compatible-mode/v1")
    p.add_argument("--api-key", default="", help="API Key")
    p.add_argument("--model", default="qwen-vl-plus")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature（默认读取 annotation_llm_temperature，通常为 0）",
    )
    p.add_argument("--repeats", type=int, default=3, help="稳定性重复次数")
    p.add_argument("--skip-llm", action="store_true", help="只导出裁剪图，不调 API")
    p.add_argument(
        "--compare-concurrency",
        action="store_true",
        help="额外跑 concurrency=1 对比",
    )
    p.add_argument(
        "--legacy-full-pool",
        action="store_true",
        help="阶段 4 额外输出「全量候选 baseline」对比（旧行为）",
    )
    p.add_argument(
        "--batch-quick",
        action="store_true",
        help="批量模式：跳过整图问答，稳定性只测 1 次（加快扫目录）",
    )
    p.add_argument(
        "--batch-quiet",
        action="store_true",
        help="批量模式：减少单图控制台输出（仍输出汇总表）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_llm and (not args.base_url or not args.api_key):
        print("错误: 需要 --base-url 和 --api-key（或加 --skip-llm 只导出裁剪图）")
        sys.exit(1)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
