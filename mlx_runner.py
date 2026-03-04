#!/usr/bin/env python3
"""
mlx_runner.py — CorpusCrafter MLX Edition
------------------------------------------
Processes a PDF into an LLM fine-tuning dataset using a local
MLX model.

    python mlx_runner.py \\
        --pdf /path/to/file.pdf \\
        --output ./output \\
        --model mlx-community/gemma-3-4b-it-qat-4bit \\
        --chunk-size 1500 \\
        --chunk-overlap 150 \\
        --language en
"""

import os
import sys
import re
import json
import uuid
import time
import logging
import unicodedata
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from tqdm import tqdm
import pandas as pd
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("mlx_runner.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── language detection ─────────────────────────────────────────────
try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ── MLX availability ─────────────────────────────────────────────────────────
try:
    import mlx_lm
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract clean text from a PDF, skipping blank/image pages and footnotes."""
    stats: Dict[str, Any] = {
        "total_pages": 0,
        "skipped_pages": 0,
        "empty_pages": 0,
        "image_pages": 0,
    }
    extracted = ""

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        stats["total_pages"] = len(reader.pages)

        start = min(1, stats["total_pages"] - 1)
        end   = max(start, stats["total_pages"] - 2)

        for i in range(start, end):
            page = reader.pages[i]
            page_text = page.extract_text() or ""

            if not page_text.strip():
                stats["empty_pages"] += 1
                stats["skipped_pages"] += 1
                continue

            if len(page_text.split()) < 20:
                stats["image_pages"] += 1
                stats["skipped_pages"] += 1
                continue

            filtered_lines = []
            for line in page_text.split("\n"):
                # skip short numbered lines (footnotes / captions)
                if re.match(r"^\d+\s", line) and len(line) < 100:
                    continue
                if len(line.strip()) < 5:
                    continue
                filtered_lines.append(line)

            extracted += "\n".join(filtered_lines) + "\n\n"

    logger.info(
        f"Extracted text from {stats['total_pages']} pages "
        f"({stats['skipped_pages']} skipped)."
    )
    return extracted, stats


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_language(text: str, hint: str = "auto") -> str:
    if hint != "auto":
        return hint
    if not LANGDETECT_AVAILABLE:
        return "en"
    try:
        lang = langdetect.detect(text[:2000])
        return lang if lang in ("en", "tr") else "en"
    except Exception:
        return "en"


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

# Language-specific heading/question patterns
_LANG_CFG = {
    "en": {
        "heading_patterns": [
            r"^[A-Z0-9]",
            r"^[IVX]+\.",
            r"^\d+\.\s",
            r"^[A-Za-z]+\s\d+",
            r"^[A-Z\s]+$",
        ],
        "question_starters": [
            "what ", "why ", "how ", "which ", "who ", "whom ", "whose ",
            "where ", "when ", "can ", "could ", "would ", "should ", "is ",
            "are ", "do ", "does ", "did ", "have ", "has ", "had ",
        ],
    },
    "tr": {
        "heading_patterns": [
            r"^[A-Z0-9]",
            r"^[IVX]+\.",
            r"^\d+\.\s",
            r"^[A-Za-z]+\s\d+",
            r"^[A-Z\s]+$",
            r"^[A-Za-z]+\s\d+:",
        ],
        "question_starters": [
            "ne ", "neden ", "niçin ", "niye ", "nasıl ", "hangi ", "kim ",
            "kime ", "kimi ", "nerede ", "ne zaman ", "kaç ", "mi ", "mı ",
            "mu ", "mü ",
        ],
    },
}


def _is_heading(line: str, lang_cfg: dict) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if "." in line[:-1] or "?" in line:
        return False
    return any(re.match(p, line) for p in lang_cfg["heading_patterns"])


def _is_question(line: str, lang_cfg: dict) -> bool:
    line = line.strip()
    if not line:
        return False
    if line.endswith("?"):
        return True
    lower = line.lower()
    return any(lower.startswith(s) for s in lang_cfg["question_starters"])


def preprocess_text(text: str, language: str = "en") -> str:
    """Normalize, strip headings/questions, and join paragraphs."""
    lang_cfg = _LANG_CFG.get(language, _LANG_CFG["en"])
    text = unicodedata.normalize("NFKC", text)
    lines = text.split("\n")

    processed_paragraphs: List[str] = []
    current_paragraph: List[str] = []
    in_content = False

    for line in lines:
        line = line.strip()
        if not line:
            if current_paragraph:
                processed_paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            in_content = False
            continue
        if _is_heading(line, lang_cfg):
            if current_paragraph:
                processed_paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            in_content = True
            continue
        if _is_question(line, lang_cfg):
            continue
        if in_content and len(line) >= 20:
            current_paragraph.append(line)

    if current_paragraph:
        processed_paragraphs.append(" ".join(current_paragraph))

    result = "\n\n".join(processed_paragraphs)
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\n\s*\n", "\n\n", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[dict]:
    """Split text into overlapping chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents([text])
    chunks = []
    for doc in docs:
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": doc.page_content,
        })
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# MLX QUESTION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class MLXQuestionGenerator:
    """Generates a question per chunk via a local MLX model (lazy-loaded)."""

    SYSTEM_EN = "You are a helpful educational assistant. You create thoughtful and meaningful questions about given texts."
    SYSTEM_TR = "Sen yardımcı bir eğitim asistanısın. Verilen metinler hakkında düşündürücü ve anlamlı sorular oluşturursun."

    def __init__(
        self,
        model_id: str = "mlx-community/gemma-3-4b-it-qat-4bit",
        max_tokens: int = 120,
        temperature: float = 0.6,
        system_prompt: Optional[str] = None,
    ):
        self.model_id   = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._model     = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        if not MLX_AVAILABLE:
            raise ImportError("mlx-lm is not installed. Run: pip install mlx-lm")
        logger.info(f"Loading MLX model: {self.model_id}  (downloads once if not cached)")
        self._model, self._tokenizer = mlx_lm.load(self.model_id)
        logger.info("Model loaded ✓")

    def generate(self, chunk_text: str, language: str = "en") -> str:
        self._load()

        system_msg = self.system_prompt or (
            self.SYSTEM_TR if language == "tr" else self.SYSTEM_EN
        )
        if language == "tr":
            user_content = (
                "Aşağıdaki metni oku ve bu metin hakkında en alakalı, açık uçlu tek bir soru oluştur.\n"
                "Soru, metindeki ana fikri veya önemli bir kavramı anlamaya yönelik olmalıdır.\n\n"
                f"Metin:\n{chunk_text.strip()}\n\nSoru:"
            )
        else:
            user_content = (
                "Read the following text and create a single, relevant, open-ended question about it.\n"
                "The question should aim to understand the main idea or an important concept in the text.\n\n"
                f"Text:\n{chunk_text.strip()}\n\nQuestion:"
            )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_content},
        ]

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"[SYSTEM]{system_msg}[/SYSTEM]\n[USER]{user_content}[/USER]\n[ASSISTANT]"

        # In newer versions of mlx-lm (0.30+), kwargs are strictly typed.
        # It's safest to avoid passing `temp` or `temperature` and stick to standard params.
        response = mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )

        response = response.strip()
        # Qwen3 often emits reasoning inside <think> tags. Strip them for clean datasets.
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        
        return response


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_dataset(rows: List[dict], output_dir: str, base_name: str, model_id: str):
    os.makedirs(output_dir, exist_ok=True)
    slug = model_id.replace("/", "_").replace("-", "_")
    stem = f"{base_name}_{slug}"

    # CSV
    csv_path = os.path.join(output_dir, f"{stem}_dataset.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"CSV  → {csv_path}")

    # JSONL
    jsonl_path = os.path.join(output_dir, f"{stem}_dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            messages = []
            if row.get("system"):
                messages.append({"role": "system", "content": row["system"]})
            messages.append({"role": "user",      "content": row["user"]})
            messages.append({"role": "assistant", "content": row["assistant"]})
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    logger.info(f"JSONL→ {jsonl_path}")

    return csv_path, jsonl_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    pdf_path = os.path.abspath(args.pdf)
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"\nPDF    : {pdf_path}")
    print(f"Model  : {args.model if not args.dry_run else '(dry-run — no model)'}")
    print(f"Chunks : size={args.chunk_size}  overlap={args.chunk_overlap}")
    print()

    # 1. Extract
    text, stats = extract_text_from_pdf(pdf_path)
    language = detect_language(text, args.language)

    print(f"Pages  : {stats['total_pages']} total | {stats['skipped_pages']} skipped")
    print(f"Language: {language}")

    # 2. Preprocess
    processed = preprocess_text(text, language)

    # 3. Chunk
    chunks = chunk_text(processed, args.chunk_size, args.chunk_overlap)
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    print(f"🔪  Chunks : {len(chunks)}\n")

    # 4. Dry-run — just show previews
    if args.dry_run:
        print("── Dry-run: chunk previews ──────────────────────────────────────\n")
        for i, chunk in enumerate(chunks):
            preview = chunk["text"][:220].replace("\n", " ")
            print(f"  [{i+1:>4}] {preview}…")
        print("\nDry-run complete — no model was loaded.")
        return

    # 5. Generate questions
    generator = MLXQuestionGenerator(
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
    )

    rows = []
    for chunk in tqdm(chunks, desc="Generating questions", unit="chunk"):
        try:
            question = generator.generate(chunk["text"], language=language)
        except Exception as e:
            logger.error(f"Generation error for chunk {chunk['chunk_id']}: {e}")
            question = "Could not generate question."

        rows.append({
            "chunk_id":  chunk["chunk_id"],
            "system":    args.system_prompt or "",
            "user":      chunk["text"],
            "assistant": question,
        })

    # 6. Save
    base_name = Path(pdf_path).stem
    csv_path, jsonl_path = save_dataset(rows, args.output, base_name, args.model)

    print(f"\nDone! {len(rows)} Q&A pairs generated.")
    print(f"   CSV  → {csv_path}")
    print(f"   JSONL→ {jsonl_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="CorpusCrafter MLX — PDF to fine-tuning dataset, locally on Apple Silicon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdf",    required=True, help="Path to input PDF.")
    p.add_argument("--output", default="output", help="Output directory.")

    p.add_argument("--model", "-m",
        default="mlx-community/gemma-3-4b-it-qat-4bit",
        help="HuggingFace mlx-community model ID.",
    )
    p.add_argument("--max-tokens",  type=int,   default=120,  help="Max tokens per generated question.")
    p.add_argument("--temperature", type=float, default=0.6,  help="Sampling temperature.")
    p.add_argument("--system-prompt", default=None, help="Override built-in system prompt.")

    p.add_argument("--chunk-size",    type=int, default=1500, help="Characters per chunk.")
    p.add_argument("--chunk-overlap", type=int, default=150,  help="Overlap between chunks.")
    p.add_argument("--language", "-l", default="auto", help="en | tr | auto (auto-detect).")

    p.add_argument("--max-chunks", type=int, default=None,
        help="Process only the first N chunks (useful for quick tests).")
    p.add_argument("--dry-run", action="store_true",
        help="Chunk only — do not load the model or generate questions.")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
