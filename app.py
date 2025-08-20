import os
import re
from io import BytesIO
from urllib.parse import urlparse, urljoin, unquote

import requests
import streamlit as st
from bs4 import BeautifulSoup

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

def get_session(referer: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    })
    if referer:
        s.headers["Referer"] = referer
    return s

def fetch_html(page_url: str) -> str:
    with get_session("https://www.alibaba.com/") as s:
        r = s.get(page_url, timeout=25, allow_redirects=True)
        r.raise_for_status()
        return r.text

def absolutize(u: str, base: str) -> str:
    return u if urlparse(u).netloc else urljoin(base, u)

def _unescape_js_url(u: str) -> str:
    return u.replace("\\/", "/")

def extract_video_urls_from_html(html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    found: list[str] = []

    for v in soup.select("div.react-dove-video video"):
        if v.get("src"):
            found.append(v["src"])
        for s in v.select("source[src]"):
            found.append(s["src"])

    if not found:
        for v in soup.find_all("video"):
            if v.get("src"):
                found.append(v["src"])
            for s in v.find_all("source"):
                if s.get("src"):
                    found.append(s["src"])

    if not found:
        for m in re.findall(r'https?://[^\s"\'<>]+?\.mp4[^\s"\'<>]*', html, flags=re.I):
            found.append(m)
        for m in re.findall(r'https?:\\?/\\?[^"\'<>]+?\.mp4[^"\'<>]*', html, flags=re.I):
            found.append(_unescape_js_url(m))

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

def guess_filename_from_url(video_url: str) -> str:
    name = os.path.basename(urlparse(video_url).path) or "video"
    name = unquote(re.sub(r"[^\w\-. ]+", "_", name)).strip("._ ") or "video"
    if not name.lower().endswith((".mp4", ".mov", ".m4v", ".webm")):
        name += ".mp4"
    return name

def fetch_video_bytes(video_url: str, page_url: str, max_bytes: int = 800 * 1024 * 1024) -> BytesIO:

    buf = BytesIO()
    with get_session(page_url) as s:
        with s.get(video_url, stream=True, timeout=90) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 18):
                if not chunk:
                    continue
                buf.write(chunk)
                if buf.tell() > max_bytes:
                    raise RuntimeError("파일이 너무 큽니다. 메모리 다운로드 한도를 초과했습니다.")
    buf.seek(0)
    return buf

st.set_page_config(page_title="Alibaba Video Downloader", page_icon="🎬")
st.title("Alibaba Video Downloader")
st.caption("① 찾기 → ② 미리보기 → ③ 다운로드 순서로 진행합니다.")

url = st.text_input("상품 URL 을 복사 & 붙여넣기 해주세요.")

if st.button("비디오 검색"):
    if not url.strip():
        st.error("URL 을 입력하여 주십시오.")
    else:
        try:
            html = fetch_html(url.strip())
            vids = extract_video_urls_from_html(html, url.strip())
        except Exception as e:
            vids = []
            st.error(f"가져오기/파싱 오류: {e}")

        if not vids:
            st.warning("검색이 되는 비디오가 존재하지 않습니다.")
            st.session_state.pop("vid_urls", None)
            st.session_state.pop("chosen_vid", None)
            st.session_state.pop("dl_data", None)
            st.session_state.pop("dl_name", None)
        else:
            st.session_state["vid_urls"] = vids
            st.session_state["chosen_vid"] = vids[0]
            st.session_state.pop("dl_data", None)
            st.session_state.pop("dl_name", None)
            st.success(f"비디오 링크가 {len(vids)}개 발견되었습니다.")

if "vid_urls" in st.session_state:
    st.selectbox(
        "비디오 파일 선택",
        st.session_state["vid_urls"],
        key="chosen_vid",
        on_change=lambda: (st.session_state.pop("dl_data", None),
                           st.session_state.pop("dl_name", None)),
    )

    with st.expander("미리보기", expanded=True):
        st.video(st.session_state["chosen_vid"])

        if st.button("비디오 다운로드"):
            try:
                data = fetch_video_bytes(st.session_state["chosen_vid"], url.strip())
                st.session_state["dl_data"] = data
                st.session_state["dl_name"] = guess_filename_from_url(st.session_state["chosen_vid"])
                st.success("다운로드 준비가 완료되었습니다.")
            except Exception as e:
                st.session_state.pop("dl_data", None)
                st.session_state.pop("dl_name", None)
                st.error(f"다운로드 실패: {e}")

        if "dl_data" in st.session_state and "dl_name" in st.session_state:
            st.download_button(
                label="파일 저장",
                data=st.session_state["dl_data"],
                file_name=st.session_state["dl_name"],
                mime="video/mp4",
            )