# src/textproc.py
# 공통 전처리 + 안전한 문장 분할 + 토큰 카운트

import re
from typing import List

import unicodedata

# 종결부호(가변 길이 포함) + 닫힘기호 + 공백/개행을 구분자로 사용
# lookbehind 미사용 → "Alternation alternatives ..." 오류 회피
SENT_END_SPLIT = re.compile(
    r'([\.!\?]|…|[！？]|\.{2,}|[!?]{2,})[\"”’)\]\}»›]*[\s\r\n]+'
)


def clean_text(text: str) -> str:
    """학습/예측 공용 전처리: 유니코드 정규화, 개행 정리, 공백 축소, 제어문자 제거"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # 줄바꿈을 문장 경계로 유지하되, 분할을 위해 공백으로 통일
    text = re.sub(r'[\r\n]+', ' ', text)
    # 연속 공백 축소
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # 비인쇄 제어문자 제거
    text = ''.join(ch for ch in text if ch.isprintable())
    return text


def split_sentences(text: str) -> List[str]:
    """종결부호를 보존하면서 문장을 분리"""
    if not text:
        return []
    text = clean_text(text)
    parts = SENT_END_SPLIT.split(text)
    # parts = [chunk0, term0, chunk1, term1, ..., last_chunk]
    sents: List[str] = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        term = parts[i + 1] if i + 1 < len(parts) else ""
        sent = (chunk + term).strip()
        if sent:
            sents.append(sent)
        i += 2
    # 마지막에 종결부호 없이 남은 chunk가 있을 수 있음
    if len(parts) % 2 == 1:
        last_chunk = parts[-1].strip()
        if last_chunk:
            sents.append(last_chunk)
    return sents


def split_to_under_max_tokens(text: str, tokenizer, max_len: int) -> List[str]:
    """
    문장 단위로 누적하여 max_len-2 토큰(budget) 이하 청크 리스트 생성
    (CLS/SEP 고려해 -2 권장). 문장 하나가 이미 budget을 넘으면 그대로 둠
    → 토크나이저 truncation에 맡김(학습 손실 최소화).
    """
    if not text:
        return [""]
    sents = split_sentences(text)
    chunks: List[str] = []
    cur = ""
    budget = max_len - 2

    for s in sents if sents else [clean_text(text)]:
        cand = (cur + " " + s).strip() if cur else s
        if len(tokenizer.tokenize(cand)) <= budget:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
                cur = s
            else:
                # 문장 자체가 너무 길면 그대로 추가(이후 truncation으로 처리)
                chunks.append(s)
                cur = ""

    if cur:
        chunks.append(cur)
    return chunks or [clean_text(text)]


def count_tokens(text: str, tokenizer) -> int:
    """전처리 후 토큰 개수"""
    return len(tokenizer.tokenize(clean_text(text)))
