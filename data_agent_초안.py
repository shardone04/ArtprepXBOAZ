"""
art_admission_crawler_agent.py

미대 입시 합격작 데이터 수집 Agent 임시 코드

역할:
1. 검색 대상 URL 목록을 입력받는다.
2. 각 페이지에서 이미지와 주변 텍스트를 추출한다.
3. 이미지 파일을 다운로드한다.
4. 이미지 해시, perceptual hash, 크기, 품질 점수를 계산한다.
5. 페이지 텍스트에서 학교/학과/전형/유형/연도 라벨을 임시 추정한다.
6. 권리 상태를 임시 추정한다.
7. CLIP으로 이미지 벡터를 생성한다.
8. JSONL 형태의 검수 큐 데이터를 저장한다.

주의:
- 인터넷 공개 이미지를 바로 학습/서비스 노출에 사용하면 안 된다.
- permission_status가 licensed가 아닌 데이터는 서비스 노출 금지.
- 이 코드는 MVP용 초안이며, 실제 배포 전 robots.txt, 사이트 약관, 법무 검토가 필요하다.
"""

import os
import re
import json
import uuid
import time
import hashlib
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageFilter
import imagehash

import torch
from transformers import CLIPProcessor, CLIPModel


# =========================
# 기본 설정
# =========================

OUTPUT_DIR = "./collected_artworks"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
JSONL_PATH = os.path.join(OUTPUT_DIR, "review_queue.jsonl")

os.makedirs(IMAGE_DIR, exist_ok=True)

USER_AGENT = "ArtAdmissionResearchBot/0.1 contact: your-email@example.com"

REQUEST_HEADERS = {
    "User-Agent": USER_AGENT
}

CRAWL_DELAY_SECONDS = 1.0

MIN_WIDTH = 300
MIN_HEIGHT = 300


# =========================
# 임시 키워드 사전
# 실제 서비스에서는 관리자 태그 사전으로 분리 추천
# =========================

SCHOOL_KEYWORDS = [
    "홍익대", "홍익대학교",
    "국민대", "국민대학교",
    "서울대", "서울대학교",
    "이화여대", "이화여자대학교",
    "건국대", "건국대학교",
    "한양대", "한양대학교",
    "중앙대", "중앙대학교",
    "성신여대", "성신여자대학교",
    "숙명여대", "숙명여자대학교",
    "서울과기대", "서울과학기술대학교",
]

DEPARTMENT_KEYWORDS = [
    "디자인학부",
    "시각디자인",
    "산업디자인",
    "공업디자인",
    "금속공예",
    "도예",
    "회화",
    "서양화",
    "동양화",
    "조소",
    "애니메이션",
    "영상디자인",
    "패션디자인",
]

ADMISSION_TYPE_KEYWORDS = [
    "수시",
    "정시",
    "실기",
    "학생부",
    "특기자",
]

EXAM_TYPE_KEYWORDS = [
    "기초디자인",
    "기초조형",
    "발상과표현",
    "사고의전환",
    "인체수채화",
    "정물수채화",
    "소묘",
    "상황표현",
    "칸만화",
]


# =========================
# 데이터 스키마
# =========================

@dataclass
class ArtworkAsset:
    asset_id: str
    image_url: str
    source_page_url: str
    source_site_name: str
    crawl_datetime: str

    local_image_path: str
    image_sha256: str
    perceptual_hash: str
    width: int
    height: int
    image_format: str

    raw_context_text: str

    auto_labels: Dict
    label_confidence: Dict

    permission_status: str
    copyright_notes: str

    quality_score: float
    duplicate_group_id: Optional[str]

    visual_tags: Dict
    generated_caption: str

    image_embedding: List[float]
    embedding_model: str
    embedding_version: str

    review_status: str
    review_notes: str


# =========================
# CLIP 임베딩 모델 로드
# =========================

class ImageEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()

    @torch.no_grad()
    def embed_image(self, image_path: str) -> List[float]:
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(
            p=2,
            dim=-1,
            keepdim=True
        )

        return image_features[0].cpu().tolist()


# =========================
# 크롤링 유틸
# =========================

def fetch_html(url: str) -> str:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
    response.raise_for_status()
    return response.text


def get_site_name(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_page_text(soup: BeautifulSoup) -> str:
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    meta_desc = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        meta_desc = meta["content"]

    headings = " ".join(
        h.get_text(" ", strip=True)
        for h in soup.find_all(["h1", "h2", "h3"])
    )

    body_text = soup.get_text(" ", strip=True)

    combined = f"{title} {meta_desc} {headings} {body_text}"
    return normalize_text(combined)[:5000]


def extract_image_candidates(page_url: str, soup: BeautifulSoup) -> List[Dict]:
    candidates = []

    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-original")
            or img.get("data-lazy")
        )

        if not src:
            continue

        image_url = urljoin(page_url, src)

        alt = img.get("alt", "")
        title = img.get("title", "")

        parent_text = ""
        parent = img.find_parent()
        if parent:
            parent_text = parent.get_text(" ", strip=True)

        context = normalize_text(
            f"{alt} {title} {parent_text}"
        )

        candidates.append({
            "image_url": image_url,
            "context_text": context
        })

    return candidates


def is_probably_artwork_image(image_url: str, context_text: str) -> bool:
    lower_url = image_url.lower()
    lower_context = context_text.lower()

    blocked_patterns = [
        "logo",
        "icon",
        "sprite",
        "banner",
        "profile",
        "avatar",
        "kakao",
        "naver",
        "facebook",
        "instagram",
    ]

    if any(pattern in lower_url for pattern in blocked_patterns):
        return False

    positive_keywords = [
        "합격",
        "합격작",
        "재현작",
        "우수작",
        "실기",
        "기초디자인",
        "기초조형",
        "발상과표현",
        "사고의전환",
        "입시미술",
        "미대입시",
        "디자인",
        "소묘",
        "수채화",
    ]

    if any(keyword in context_text for keyword in positive_keywords):
        return True

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    return any(lower_url.split("?")[0].endswith(ext) for ext in image_extensions)


def download_image(image_url: str, asset_id: str) -> Optional[str]:
    try:
        response = requests.get(
            image_url,
            headers=REQUEST_HEADERS,
            timeout=20
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "image" not in content_type:
            return None

        ext = "jpg"
        if "png" in content_type:
            ext = "png"
        elif "webp" in content_type:
            ext = "webp"
        elif "jpeg" in content_type or "jpg" in content_type:
            ext = "jpg"

        local_path = os.path.join(IMAGE_DIR, f"{asset_id}.{ext}")

        with open(local_path, "wb") as f:
            f.write(response.content)

        return local_path

    except Exception as e:
        print(f"[이미지 다운로드 실패] {image_url} / {e}")
        return None


# =========================
# 이미지 분석
# =========================

def calculate_sha256(file_path: str) -> str:
    h = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def inspect_image(file_path: str) -> Dict:
    image = Image.open(file_path)
    width, height = image.size
    image_format = image.format or "unknown"

    phash = str(imagehash.phash(image))

    return {
        "width": width,
        "height": height,
        "format": image_format.lower(),
        "perceptual_hash": phash
    }


def estimate_blur_score(image_path: str) -> float:
    """
    아주 간단한 흐림 추정.
    값이 높을수록 선명하다고 간주.
    OpenCV 없이 PIL만 사용한 임시 버전.
    """
    image = Image.open(image_path).convert("L")
    edges = image.filter(ImageFilter.FIND_EDGES)

    pixels = list(edges.getdata())
    if not pixels:
        return 0.0

    return sum(pixels) / len(pixels) / 255.0


def calculate_quality_score(width: int, height: int, blur_score: float) -> float:
    size_score = min((width * height) / (1200 * 900), 1.0)
    blur_score = min(blur_score * 5, 1.0)

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return 0.2

    return round(0.65 * size_score + 0.35 * blur_score, 4)


# =========================
# 라벨 추정
# =========================

def match_keywords(text: str, keywords: List[str]) -> List[str]:
    return [kw for kw in keywords if kw in text]


def extract_year(text: str) -> Optional[int]:
    match = re.search(r"(20[0-2][0-9])", text)
    if match:
        return int(match.group(1))
    return None


def estimate_labels(raw_context_text: str) -> Dict:
    schools = match_keywords(raw_context_text, SCHOOL_KEYWORDS)
    departments = match_keywords(raw_context_text, DEPARTMENT_KEYWORDS)
    admission_types = match_keywords(raw_context_text, ADMISSION_TYPE_KEYWORDS)
    exam_types = match_keywords(raw_context_text, EXAM_TYPE_KEYWORDS)
    year = extract_year(raw_context_text)

    return {
        "school": schools[0] if schools else None,
        "department": departments[0] if departments else None,
        "admission_type": admission_types[0] if admission_types else None,
        "exam_type": exam_types[0] if exam_types else None,
        "year": year
    }


def estimate_label_confidence(labels: Dict, raw_context_text: str) -> Dict:
    confidence = {}

    for key, value in labels.items():
        if value is None:
            confidence[key] = 0.0
        elif key == "year":
            confidence[key] = 0.7
        else:
            count = raw_context_text.count(str(value))
            confidence[key] = min(0.5 + count * 0.15, 0.95)

    return confidence


# =========================
# 권리 상태 추정
# =========================

def estimate_permission_status(raw_context_text: str, image_url: str) -> Dict:
    """
    자동 판단은 법적 결론이 아니라 검수 후보 분류용.
    unknown/research_candidate/prohibited 정도만 임시 부여.
    licensed는 사람이 계약 확인 후에만 부여하는 것을 권장.
    """

    text = raw_context_text.lower()
    url = image_url.lower()

    prohibited_keywords = [
        "무단 전재",
        "무단전재",
        "재배포 금지",
        "복제 금지",
        "all rights reserved",
        "copyright",
        "저작권",
        "워터마크",
    ]

    permissive_keywords = [
        "cc-by",
        "creative commons",
        "공공누리",
        "public domain",
    ]

    if any(keyword.lower() in text for keyword in prohibited_keywords):
        return {
            "permission_status": "prohibited",
            "copyright_notes": "저작권/재배포 제한 문구 감지"
        }

    if any(keyword.lower() in text for keyword in permissive_keywords):
        return {
            "permission_status": "public_license",
            "copyright_notes": "공개 라이선스 관련 문구 감지. 조건 확인 필요"
        }

    if "watermark" in url or "logo" in url:
        return {
            "permission_status": "research_candidate",
            "copyright_notes": "이미지 URL상 워터마크/로고 가능성 있음"
        }

    return {
        "permission_status": "unknown",
        "copyright_notes": "명시적 권리 정보 없음. 사람 검수 전 사용 금지"
    }


# =========================
# 시각 태그 임시 생성
# =========================

def estimate_visual_tags(raw_context_text: str) -> Dict:
    """
    실제로는 VLM이나 관리자 검수로 만드는 게 좋음.
    여기서는 텍스트 기반 임시 태그만 생성.
    """

    composition = []
    color = []
    rendering = []
    subject = []
    style = []

    if "중앙" in raw_context_text:
        composition.append("중앙집중형")
    if "대각선" in raw_context_text:
        composition.append("대각선 구도")
    if "여백" in raw_context_text:
        composition.append("여백 강조")

    if "고채도" in raw_context_text or "강한 색" in raw_context_text:
        color.append("고채도")
    if "저채도" in raw_context_text:
        color.append("저채도")
    if "보색" in raw_context_text:
        color.append("보색 대비")

    if "정밀" in raw_context_text:
        rendering.append("정밀묘사")
    if "명암" in raw_context_text:
        rendering.append("강한 명암")
    if "그라데이션" in raw_context_text:
        rendering.append("부드러운 그라데이션")

    for item in ["금속", "유리", "천", "식물", "기하", "도형", "인체"]:
        if item in raw_context_text:
            subject.append(item)

    for item in EXAM_TYPE_KEYWORDS:
        if item in raw_context_text:
            style.append(item)

    density = "medium"
    if "밀도" in raw_context_text and "높" in raw_context_text:
        density = "high"
    elif "여백" in raw_context_text:
        density = "low"

    return {
        "composition": composition,
        "color": color,
        "density": density,
        "rendering": rendering,
        "subject": subject,
        "style": style
    }


def generate_caption_stub(labels: Dict, visual_tags: Dict) -> str:
    """
    임시 caption.
    실제로는 VLM captioning 모델 또는 LLM+검수로 대체.
    """
    parts = []

    if labels.get("school"):
        parts.append(f"{labels['school']} 관련")
    if labels.get("exam_type"):
        parts.append(labels["exam_type"])
    if visual_tags.get("style"):
        parts.append(", ".join(visual_tags["style"]))

    if not parts:
        return "미대 입시 합격작 후보 이미지"

    return " / ".join(parts) + " 작품 후보"


# =========================
# 메인 Agent
# =========================

class ArtAdmissionCrawlerAgent:
    def __init__(self):
        self.embedder = ImageEmbedder()

    def crawl_page(self, page_url: str) -> List[ArtworkAsset]:
        print(f"[크롤링 시작] {page_url}")

        html = fetch_html(page_url)
        soup = BeautifulSoup(html, "html.parser")

        page_text = extract_page_text(soup)
        image_candidates = extract_image_candidates(page_url, soup)

        results = []

        for candidate in image_candidates:
            image_url = candidate["image_url"]
            image_context = candidate["context_text"]

            raw_context_text = normalize_text(
                f"{page_text} {image_context}"
            )[:5000]

            if not is_probably_artwork_image(image_url, raw_context_text):
                continue

            asset_id = str(uuid.uuid4())
            local_path = download_image(image_url, asset_id)

            if not local_path:
                continue

            try:
                image_info = inspect_image(local_path)

                width = image_info["width"]
                height = image_info["height"]

                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    print(f"[제외: 이미지 작음] {image_url}")
                    continue

                sha256 = calculate_sha256(local_path)
                blur_score = estimate_blur_score(local_path)
                quality_score = calculate_quality_score(width, height, blur_score)

                labels = estimate_labels(raw_context_text)
                label_confidence = estimate_label_confidence(labels, raw_context_text)

                permission = estimate_permission_status(
                    raw_context_text,
                    image_url
                )

                visual_tags = estimate_visual_tags(raw_context_text)
                caption = generate_caption_stub(labels, visual_tags)

                image_embedding = self.embedder.embed_image(local_path)

                review_status = "pending"

                if permission["permission_status"] in ["prohibited", "unknown"]:
                    review_status = "needs_legal_review"

                asset = ArtworkAsset(
                    asset_id=asset_id,
                    image_url=image_url,
                    source_page_url=page_url,
                    source_site_name=get_site_name(page_url),
                    crawl_datetime=datetime.datetime.now(
                        datetime.timezone(datetime.timedelta(hours=9))
                    ).isoformat(),

                    local_image_path=local_path,
                    image_sha256=sha256,
                    perceptual_hash=image_info["perceptual_hash"],
                    width=width,
                    height=height,
                    image_format=image_info["format"],

                    raw_context_text=raw_context_text,

                    auto_labels=labels,
                    label_confidence=label_confidence,

                    permission_status=permission["permission_status"],
                    copyright_notes=permission["copyright_notes"],

                    quality_score=quality_score,
                    duplicate_group_id=None,

                    visual_tags=visual_tags,
                    generated_caption=caption,

                    image_embedding=image_embedding,
                    embedding_model=self.embedder.model_name,
                    embedding_version="clip-vit-base-patch32__v0.1",

                    review_status=review_status,
                    review_notes=""
                )

                results.append(asset)

            except Exception as e:
                print(f"[이미지 처리 실패] {image_url} / {e}")
                continue

            time.sleep(CRAWL_DELAY_SECONDS)

        return results

    def crawl_urls(self, urls: List[str]) -> List[ArtworkAsset]:
        all_assets = []

        for url in urls:
            try:
                assets = self.crawl_page(url)
                all_assets.extend(assets)
            except Exception as e:
                print(f"[페이지 실패] {url} / {e}")

        return all_assets

    def save_jsonl(self, assets: List[ArtworkAsset], path: str = JSONL_PATH):
        with open(path, "a", encoding="utf-8") as f:
            for asset in assets:
                f.write(
                    json.dumps(
                        asdict(asset),
                        ensure_ascii=False
                    ) + "\n"
                )

        print(f"[저장 완료] {len(assets)}개 asset → {path}")


# =========================
# 실행 예시
# =========================

if __name__ == "__main__":
    seed_urls = [
        # 여기에 수집 대상 페이지 URL을 넣으면 됨
        # 예:
        # "https://example-academy.com/pass-artworks/2024",
        # "https://example-school.com/admission/artworks",
    ]

    agent = ArtAdmissionCrawlerAgent()
    assets = agent.crawl_urls(seed_urls)
    agent.save_jsonl(assets)