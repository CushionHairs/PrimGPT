# web_utils.py
import logging
from typing import Optional, Tuple

import requests
from newspaper import Article, ArticleException
from bs4 import BeautifulSoup
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.chrome.options import Options 
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st
import time 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
REQUEST_TIMEOUT = 15 # seconds
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" # Common user agent

def sanitize_text(text: str) -> str:
    return text.replace("~", "&#126;")

def fetch_main_content(url: str) -> Tuple[Optional[str], int]:
    try:
        headers = {"User-Agent": USER_AGENT}
        article = Article(url, request_timeout=REQUEST_TIMEOUT, headers=headers)
        article.download()
        article.parse()

        main_text = sanitize_text(article.text)
        title = sanitize_text(article.title)

        if not main_text:
            logger.warning(f"Newspaper3k could not extract text from URL: {url}")
            st.warning("Could not extract main content from the webpage.")
            return None, 204  # No Content

        logger.info(f"Successfully extracted content from: {url}")
        output = f"### 📰 **Title:** {title}\n\n---\n\n📜 **Content:**\n{main_text}"
        return output, 200

    except ArticleException as e:
        logger.error(f"Newspaper3k failed for URL {url}: {e}")
        st.error(f"Failed to process the webpage: {e}")
        return None, 500
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching URL {url}: {e}")
        st.error(f"Could not connect to the URL: {e}")
        return None, 503
    except Exception as e:
        logger.exception(f"Unexpected error fetching content from {url}: {e}")
        st.error(f"An unexpected error occurred: {e}")
        return None, 500
    
def fetch_main_content_bs4(url: str) -> Tuple[Optional[str], int]:
    try:
        # Set request headers
        headers = {"User-Agent": USER_AGENT}

        # Fetch webpage content
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else 'No Title Found'
        title = sanitize_text(title)

        # Attempt to extract main content
        # First attempt: Look for <article> tags, common for news/blogs
        article_tag = soup.find('article')
        if article_tag:
            paragraphs = article_tag.find_all('p')
        else:
            # Fallback: Extract content from all <p> tags within <body>
            paragraphs = soup.body.find_all('p') if soup.body else []

        # Combine paragraphs into readable text
        main_text = '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        main_text = sanitize_text(main_text)

        if not main_text.strip():
            logger.warning(f"BeautifulSoup could not extract meaningful text from URL: {url}")
            st.warning("Could not extract main content from the webpage.")
            return None, 204  # No Content

        logger.info(f"Successfully extracted content using BeautifulSoup from: {url}")

        output = f"### 📰 **Title:** {title}\n\n---\n\n📜 **Content:**\n{main_text}"
        return output, 200

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching URL {url}: {e}")
        st.error(f"Could not connect to the URL: {e}")
        return None, 503

    except Exception as e:
        logger.exception(f"Unexpected error fetching content from {url}: {e}")
        st.error(f"An unexpected error occurred: {e}")
        return None, 500
    
def fetch_main_content_selenium(url: str, wait_time: int = 5) -> Tuple[Optional[str], int]:
    try:
        # Selenium 옵션 설정 (headless 모드 사용, 브라우저 창 띄우지 않음)
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("window-size=1920,1080")
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                                    'Chrome/119.0.0.0 Safari/537.36')

        # 웹드라이버 초기화 (자동 설치)
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # 웹 페이지 접속
        logger.info(f"Fetching URL with Selenium: {url}")
        driver.get(url)

        # 페이지 로딩 대기 (필요한 경우 wait_time 조정 가능)
        time.sleep(wait_time)  # 동적 콘텐츠 로딩을 위해 충분히 기다려줍니다.

        # 페이지 소스 가져오기
        html = driver.page_source

        # BeautifulSoup으로 파싱
        soup = BeautifulSoup(html, 'html.parser')

        # 타이틀 추출하기
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else "No Title Found"
        title = sanitize_text(title)

        # 메인 콘텐츠 추출하기 (<article> 태그 우선)
        article_tag = soup.find('article')
        if article_tag:
            paragraphs = article_tag.find_all('p')
        else:
            # <article> 태그 없으면 <body> 내 모든 <p> 태그 이용
            paragraphs = soup.body.find_all('p') if soup.body else []

        # 본문 텍스트 추출
        main_text = '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        main_text = sanitize_text(main_text)
        
        # 본문이 빈 경우 처리
        if not main_text.strip():
            logger.warning(f"Selenium could not extract meaningful text from URL: {url}")
            st.warning("Could not extract main content from the webpage using Selenium.")
            return None, 204  # No Content

        # 성공적으로 콘텐츠 가져왔을 때
        output = f"### 📰 **Title:** {title}\n\n---\n\n📜 **Content:**\n{main_text}"
        logger.info(f"Successfully extracted content using Selenium from: {url}")
        return output, 200

    except Exception as e:
        logger.exception(f"Unexpected error fetching content with Selenium from {url}: {e}")
        st.error(f"An unexpected error occurred: {e}")
        return None, 500

    finally:
        # 드라이버 종료
        driver.quit()