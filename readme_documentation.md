# 🎵 음악 기반 앨범 아트 생성기

음악 파일을 업로드하면 자동으로 장르와 분위기를 분석하여 어울리는 앨범 아트를 AI로 생성하는 시스템입니다.

## 📋 프로젝트 개요

### 시스템 구성
- **로컬 서버**: FastAPI + Essentia (음악 분석)
- **Google Colab**: Stable Diffusion (이미지 생성) + Gradio (웹 인터페이스)  
- **연결**: ngrok 터널링을 통한 HTTP API 통신

### 주요 기능
- 🎼 **음악 분석**: Essentia 기반 장르 및 분위기 자동 분석
- 🎨 **이미지 생성**: Stable Diffusion v1.5로 512x512 앨범 아트 생성
- 🌐 **웹 인터페이스**: Gradio 기반 사용자 친화적 UI
- 🔗 **분산 처리**: 로컬 분석 + 클라우드 생성의 하이브리드 구조

## 🚀 빠른 시작

### Colab 노트북 사용 (권장)
가장 간단한 방법은 준비된 Colab 노트북을 사용하는 것입니다:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E4iIwQauk58iOQulB4sUgZFZLUxAfz8p?usp=sharing)

## 📖 상세 설치 가이드

### 1단계: 로컬 환경 설정

#### 필수 요구사항
- Python 3.8+
- CUDA 지원 GPU (권장)
- 최소 8GB RAM
- 인터넷 연결

#### Essentia 설치
```bash
# Ubuntu/Linux
pip install essentia-tensorflow

# Windows (WSL 권장)
# WSL Ubuntu에서 위 명령어 실행

# macOS
pip install essentia-tensorflow
```

#### 의존성 패키지 설치
```bash
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install numpy==1.24.4
pip install requests==2.31.0
```

### 2단계: 모델 파일 준비

로컬 서버용 Essentia 모델 파일들을 `./dependencies/` 디렉토리에 다운로드:

```bash
mkdir dependencies
cd dependencies

# Discogs EfficientNet 모델 (장르 분류)
wget https://essentia.upf.edu/models/classification-heads/discogs_genre/discogs-effnet-bs64-1.pb
wget https://essentia.upf.edu/models/classification-heads/discogs_genre/discogs-effnet-bs64-1.json

# MTG Jamendo 모델 (분위기 분석)  
wget https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb
wget https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json
```

### 3단계: 로컬 서버 실행

```bash
python fastapi_music_server.py
```

서버가 성공적으로 시작되면 `http://localhost:8000`에서 실행됩니다.

### 4단계: ngrok 터널링 설정

새 터미널에서:

```bash
# ngrok 설치 (한 번만)
# https://ngrok.com/download에서 다운로드

# 터널링 시작
ngrok http 8000
```

ngrok이 제공하는 HTTPS URL을 복사해둡니다 (예: `https://abc123.ngrok-free.app`).

### 5단계: Google Colab 실행

1. [Colab 노트북](https://colab.research.google.com/drive/1E4iIwQauk58iOQulB4sUgZFZLUxAfz8p?usp=sharing) 열기
2. **첫 번째 셀**: 환경 설정 실행
3. **Runtime > Restart runtime** 클릭
4. **두 번째 셀**: 설치 확인 실행
5. **세 번째 셀**: ngrok URL 수정 후 메인 앱 실행

```python
# 마지막 셀에서 이 부분을 실제 ngrok URL로 변경
NGROK_URL = "https://your-ngrok-url-here.ngrok-free.app"
```

## 🎯 사용 방법

1. **음악 파일 업로드**: MP3, WAV, M4A, FLAC 형식 지원
2. **자동 제목 감지**: 파일명에서 곡 제목 자동 추출
3. **생성 버튼 클릭**: "🎨 앨범 아트 생성" 버튼 클릭
4. **결과 확인**: 
   - 생성된 앨범 아트 이미지
   - 음악 분석 결과 (장르, 분위기)
   - 사용된 AI 프롬프트
5. **이미지 저장**: 우클릭 → "다른 이름으로 저장"

## ⚡ 성능 및 소요 시간

- **음악 분석**: 10-30초 (로컬 서버)
- **이미지 생성**: 20-60초 (GPU 사용 시)
- **총 소요 시간**: 약 1-2분
- **지원 파일 크기**: 최대 50MB
- **권장 음악 길이**: 30초 이상 (더 정확한 분석)

## 📁 프로젝트 구조

```
album-art-generator/
├── README.md                           # 이 문서
├── fastapi_music_server.py            # 로컬 FastAPI 서버 (음악 분석)
├── final_album_art_generator.py       # Colab 통합 앱
├── dependencies/                       # Essentia 모델 파일들
│   ├── discogs-effnet-bs64-1.pb
│   ├── discogs-effnet-bs64-1.json
│   ├── mtg_jamendo_moodtheme-discogs-effnet-1.pb
│   └── mtg_jamendo_moodtheme-discogs-effnet-1.json
└── docs/                              # 추가 문서들
    ├── project_analysis_prompt.md
    └── 음악 기반 앨범 아트 생성_프로젝트_주제.pdf
```

## 🔧 기술 스택

### 음악 분석
- **Essentia**: 오디오 신호 처리 및 특징 추출
- **사전학습 모델**: 
  - Discogs EfficientNet (장르 분류 - 400개 클래스)
  - MTG Jamendo (분위기 분석 - 87개 태그)

### 이미지 생성
- **Stable Diffusion v1.5**: 텍스트-이미지 생성 모델
- **PyTorch**: 딥러닝 프레임워크
- **CUDA**: GPU 가속

### 웹 인터페이스
- **Gradio**: 사용자 인터페이스
- **FastAPI**: RESTful API 서버
- **ngrok**: 터널링 서비스
