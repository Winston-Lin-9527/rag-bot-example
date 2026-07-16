from typing import Dict

from paddleocr import PaddleOCR

from models.state import ContractState
from services.llm import LLMService
from utils.file_utils import extract_native_pdf_text
from utils import progress

# Lazy-initialised singleton so the model loads once per process
_ocr_engine: PaddleOCR | None = None


def _get_ocr() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        # PP-OCRv4 mobile models are ~4x faster than v5 server on CPU.
        # All rotation/orientation detection disabled — no autorotation.
        _ocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
        )
    return _ocr_engine


def _ocr_image(path: str) -> str:
    """PaddleOCR 3.x uses predict() which returns a generator of result dicts."""
    ocr = _get_ocr()
    try:
        # 3.x API: predict() yields result dicts per image
        results = list(ocr.predict(path))
        if not results:
            return ""
        res = results[0]
        # Result is a dict-like object; rec_texts holds the text lines
        if isinstance(res, dict):
            texts = res.get("rec_texts") or []
            return "\n".join(str(t) for t in texts if t)
        # Fallback: some versions return objects with attributes
        if hasattr(res, "rec_texts"):
            return "\n".join(str(t) for t in res.rec_texts if t)
        return ""
    except AttributeError:
        # Graceful fallback to 2.x API if somehow an older engine is used
        result = ocr.ocr(path, cls=True)  # type: ignore[attr-defined]
        if not result or result[0] is None:
            return ""
        return "\n".join(line[1][0] for line in result[0] if line and len(line) >= 2)


def _vision_extract(llm: LLMService, image_path: str) -> str:
    prompt = (
        "You are a contract document reader. "
        "Extract ALL visible text from this contract page exactly as it appears. "
        "Preserve table structure using pipe | separators. "
        "Include headers, dates, prices, party names, and all clause text. "
        "Do not summarise or omit anything."
    )
    return llm.generate(prompt, images=[image_path])


def _report(page_num: int, total: int, status: str) -> None:
    msg = f"[OCR] Page {page_num + 1}/{total}: {status}"
    print(msg, flush=True)
    progress.post(msg)


def ocr_extraction_node(state: ContractState) -> dict:
    log = list(state.get("processing_log", []))
    print("OCR extraction node invoked. Current state:", state)

    # XML / pre-filled text: skip OCR
    if state.get("full_text") or state["file_type"] == "xml":
        log.append("OCR skipped – text already available")
        return {**state, "processing_log": log, "current_step": "indexing"}

    llm = LLMService()
    raw_text_by_page: Dict[int, str] = {}

    total_pages = len(state.get("page_image_paths", []))
    for page_num, img_path in enumerate(state.get("page_image_paths", [])):
        _report(page_num, total_pages, "OCR…")
        paddle_text = _ocr_image(img_path)

        # Use vision model for sparse pages (tables, forms, stamps)
        if len(paddle_text.strip()) < 80:
            _report(page_num, total_pages, "sparse — running vision model…")
            try:
                vision_text = _vision_extract(llm, img_path)
                page_text = vision_text if len(vision_text) > len(paddle_text) else paddle_text
            except Exception:
                page_text = paddle_text
        else:
            page_text = paddle_text

        _report(page_num, total_pages, f"done ({len(page_text)} chars)")
        raw_text_by_page[page_num] = page_text

    # Merge with native PDF text layer when richer
    if state["file_type"] == "pdf":
        try:
            native = extract_native_pdf_text(state["file_path"])
            for pg, nat_text in native.items():
                existing = raw_text_by_page.get(pg, "")
                if len(nat_text.strip()) > len(existing.strip()):
                    raw_text_by_page[pg] = nat_text
        except Exception:
            pass

    full_text = "\n\n".join(
        f"--- Page {pg + 1} ---\n{txt}"
        for pg, txt in sorted(raw_text_by_page.items())
    )

    log.append(f"OCR complete: {len(raw_text_by_page)} pages → {len(full_text)} chars")
    return {
        **state,
        "raw_text_by_page": raw_text_by_page,
        "full_text": full_text,
        "processing_log": log,
        "current_step": "indexing",
    }