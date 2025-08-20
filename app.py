import os, re, tempfile, subprocess
from io import BytesIO
from urllib.parse import urlparse, urljoin, unquote

import requests
import streamlit as st
from bs4 import BeautifulSoup
import imageio_ffmpeg

# Optional CV/OCR
try:
    import cv2
    import easyocr
    OCR_OK, OCR_ERR = True, None
except Exception as e:
    OCR_OK, OCR_ERR = False, e

# ---------- HTTP ----------
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

def get_session(referer: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    })
    if referer:
        s.headers["Referer"] = referer  # 일부 CDN은 필수
    return s

def fetch_html(page_url: str) -> str:
    with get_session("https://www.alibaba.com/") as s:
        r = s.get(page_url, timeout=25, allow_redirects=True)
        r.raise_for_status()
        return r.text

# ---------- extraction ----------
def _unescape_js_url(u: str) -> str:
    return u.replace("\\/", "/")

def absolutize(u: str, base: str) -> str:
    return u if urlparse(u).netloc else urljoin(base, u)

def extract_video_urls_from_html(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    found: list[str] = []

    # 1) 스크린샷 구조
    for v in soup.select("div.react-dove-video video"):
        if v.get("src"):
            found.append(v["src"])
        for s in v.select("source[src]"):
            found.append(s["src"])

    # 2) 전역 fallback
    if not found:
        for v in soup.find_all("video"):
            if v.get("src"):
                found.append(v["src"])
            for s in v.find_all("source"):
                if s.get("src"):
                    found.append(s["src"])

    # 3) 원시 HTML에서 .mp4
    if not found:
        for m in re.findall(r'https?://[^\s"\'<>]+?\.mp4[^\s"\'<>]*', html, flags=re.I):
            found.append(m)
        for m in re.findall(r'https?:\\?/\\?[^"\'<>]+?\.mp4[^"\'<>]*', html, flags=re.I):
            found.append(_unescape_js_url(m))

    # 우선순위/중복 제거/절대화
    pri, sec = [], []
    for u in found:
        (pri if ("alicdn.com" in u or "alibaba.com" in u) else sec).append(u)
    out, seen = [], set()
    for u in pri + sec:
        uu = absolutize(u, page_url)
        if uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out

# ---------- filename ----------
def guess_filename_from_url(video_url: str) -> str:
    name = os.path.basename(urlparse(video_url).path) or "video"
    name = unquote(re.sub(r"[^\w\-. ]+", "_", name)).strip("._ ") or "video"
    if not name.lower().endswith((".mp4", ".mov", ".m4v", ".webm")):
        name += ".mp4"
    return name

# ---------- in-memory download ----------
def fetch_video_bytes(video_url: str, page_url: str, max_bytes: int = 800 * 1024 * 1024) -> BytesIO:
    buf = BytesIO()
    with get_session(page_url) as s:
        with s.get(video_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 18):
                if not chunk:
                    continue
                buf.write(chunk)
                if buf.tell() > max_bytes:
                    raise RuntimeError("파일이 너무 큽니다. 메모리 한도를 초과했습니다.")
    buf.seek(0)
    return buf

# ---------- OCR blur ----------
def blur_boxes_in_frame(frame, boxes, ksize=25):
    for (x0, y0, x1, y1) in boxes:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1]
            k = max(3, ksize | 1)
            frame[y0:y1, x0:x1] = __import__("cv2").GaussianBlur(roi, (k, k), 0)
    return frame

def ocr_detect_boxes(frame, reader, conf_th=0.45):
    results = reader.readtext(frame, detail=1)
    boxes = []
    for poly, text, conf in results:
        if conf is None or conf < conf_th:
            continue
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes

def auto_blur_bottom_text(input_bytes: BytesIO,
                          bottom_ratio=0.20,
                          sample_step=2,
                          conf_th=0.45,
                          blur_ksize=27) -> BytesIO:
    if not OCR_OK:
        raise RuntimeError(f"자동 감지 블러를 사용하려면 OpenCV/easyocr가 필요합니다: {OCR_ERR}")

    import cv2, easyocr  # ensured by OCR_OK
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.mp4"); dst = os.path.join(td, "out.mp4")
        with open(src, "wb") as f: f.write(input_bytes.getbuffer())

        cap = cv2.VideoCapture(src)
        if not cap.isOpened(): raise RuntimeError("비디오를 열 수 없습니다.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(dst, fourcc, fps, (w, h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(dst, fourcc, fps, (w, h))
            if not out.isOpened():
                cap.release(); raise RuntimeError("출력 비디오를 생성할 수 없습니다.")

        reader = easyocr.Reader(['en','ko'], gpu=False)
        band_y0 = int(h * (1.0 - bottom_ratio)); band_y1 = h
        frame_idx, cached_boxes = 0, []
        while True:
            ret, frame = cap.read()
            if not ret: break
            band = frame[band_y0:band_y1, 0:w]
            if frame_idx % sample_step == 0:
                boxes_band = ocr_detect_boxes(band, reader, conf_th=conf_th)
                cached_boxes = [(x0, y0 + band_y0, x1, y1 + band_y0) for (x0, y0, x1, y1) in boxes_band]
            frame = blur_boxes_in_frame(frame, cached_boxes, ksize=blur_ksize)
            out.write(frame); frame_idx += 1
        cap.release(); out.release()
        with open(dst, "rb") as f:
            out_bytes = BytesIO(f.read())
        out_bytes.seek(0); return out_bytes

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Alibaba 비디오 다운로더", page_icon="🎬", layout="centered")
st.title("Alibaba 비디오 다운로더")
st.caption("① 상품 페이지 URL → ② 비디오 찾기 → ③ 미리보기(가능한 경우) → ④ (옵션) 자동 블러 → ⑤ 다운로드")

page_url = st.text_input("상품 URL 을 복사 & 붙여넣기 해주세요.")

# 찾기
if st.button("비디오 찾기"):
    if not page_url.strip():
        st.error("URL 을 입력하여 주십시오.")
    else:
        try:
            html = fetch_html(page_url.strip())
            vids = extract_video_urls_from_html(html, page_url.strip())
        except Exception as e:
            vids = []
            st.error(f"가져오기/파싱 오류: {e}")

        if not vids:
            st.warning("검색이 되는 비디오가 존재하지 않습니다.")
            for k in ("vid_urls","chosen_vid","dl_data","dl_name"): st.session_state.pop(k, None)
        else:
            st.session_state["vid_urls"] = vids
            st.session_state["chosen_vid"] = vids[0]  # 직접 .mp4 URL 저장 (중요!)
            for k in ("dl_data","dl_name"): st.session_state.pop(k, None)
            st.success(f"비디오 링크가 {len(vids)}개 발견되었습니다.")
            st.write("예시 링크:", vids[0])

# 후보가 있으면 미리보기 + 옵션 + 다운로드
if "vid_urls" in st.session_state:
    st.selectbox(
        "비디오 파일 선택",
        st.session_state["vid_urls"],
        key="chosen_vid",
        help="여러 개가 발견되면 원하는 것을 선택하세요.",
    )

    # 미리보기 시도 (주의: CDN이 Referer를 요구하면 재생이 안 될 수 있음)
    with st.expander("미리보기", expanded=True):
        try:
            st.video(st.session_state["chosen_vid"])
            st.caption("재생이 안 되면 CDN이 헤더(Referer)를 요구하는 경우입니다. 아래 다운로드는 정상 동작합니다.")
        except Exception:
            st.info("이 미디어는 브라우저 내 미리보기가 제한될 수 있습니다. 아래 다운로드를 이용하세요.")

    st.markdown("### 텍스트 처리 옵션 (OCR 필요)")
    opt_auto_blur = st.checkbox("자동 감지 블러(실험적, 느림)", value=False,
                                help="하단 영역(기본 20%)에서 텍스트를 OCR로 감지해 해당 부분만 블러. Cloud에서는 설치가 무거울 수 있습니다.")
    if opt_auto_blur and not OCR_OK:
        st.warning(f"자동 블러를 사용하려면 OpenCV/easyocr가 필요합니다. (Import 실패: {OCR_ERR})")

    auto_ratio = st.slider(
        "하단 블러 비율(%)", 10, 30, 20,
        disabled=not (opt_auto_blur and OCR_OK)
    ) / 100.0
    st.caption("화면 하단 몇 %를 블러 처리할지 설정합니다.")

    auto_step = st.slider(
        "프레임 샘플링 (매 N프레임)", 1, 5, 2,
        disabled=not (opt_auto_blur and OCR_OK)
    )
    st.caption("값이 작을수록 정확하지만 느리고, 클수록 빠르지만 정확도가 떨어집니다.")

    auto_conf = st.slider(
        "텍스트 인지 신뢰도", 10, 80, 45,
        disabled=not (opt_auto_blur and OCR_OK)
    ) / 100.0
    st.caption("텍스트로 판단할 최소 신뢰도 기준입니다. 낮으면 과도한 블러 처리, 높으면 블러 누락 가능성이 있습니다.")

    auto_ksize = st.slider(
        "블러 강도", 11, 51, 27, step=2,
        disabled=not (opt_auto_blur and OCR_OK)
    )
    st.caption("적용할 블러의 세기입니다. 값이 클수록 글자가 강하게 흐려집니다.")

    if st.button("텍스트 처리 후 다운로드"):
        try:
            raw_bytes = fetch_video_bytes(st.session_state["chosen_vid"], page_url.strip())
            out_bytes = raw_bytes
            if opt_auto_blur and OCR_OK:
                out_bytes = auto_blur_bottom_text(
                    out_bytes,
                    bottom_ratio=auto_ratio,
                    sample_step=auto_step,
                    conf_th=auto_conf,
                    blur_ksize=auto_ksize,
                )
            st.download_button(
                label="파일 저장",
                data=out_bytes,
                file_name=guess_filename_from_url(st.session_state["chosen_vid"]),
                mime="video/mp4",
            )
            st.success("준비 완료. '파일 저장'을 클릭하세요.")
        except Exception as e:
            st.error(f"처리 실패: {e}")