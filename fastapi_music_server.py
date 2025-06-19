# fastapi_music_server.py
# 로컬 환경용 Essentia 기반 음악 분류 FastAPI 서버

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import time
import traceback

# FastAPI 관련
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# 음악 분석 관련
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Essentia 라이브러리 (로컬 환경에서 설치 필요)
try:
    from essentia.standard import (
        AudioLoader,
        MonoLoader,
        TensorflowPredictEffnetDiscogs,
        TensorflowPredict2D
    )
    import essentia

    ESSENTIA_AVAILABLE = True
    print("✅ Essentia 라이브러리 로드 성공")
except ImportError as e:
    print(f"❌ Essentia 라이브러리 로드 실패: {e}")
    print("해결방법: pip install essentia-tensorflow")
    ESSENTIA_AVAILABLE = False


class EssentiaServer:
    """Essentia 기반 음악 분류 서버"""

    def __init__(self, model_path: str = "./dependencies"):
        """
        초기화

        Args:
            model_path: 모델 파일들이 있는 디렉토리 경로
        """
        self.model_path = Path(model_path)
        self.genre_classes = []
        self.mood_classes = []

        # 모델 관련 변수들
        self.embeddings_model = None
        self.mood_model = None
        self.genre_model = None

        # 분위기 태그 카테고리
        self.mood_tags = []
        self.theme_tags = []
        self.function_tags = []

        # 초기화
        self.is_ready = False
        if ESSENTIA_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """모델 및 메타데이터 초기화"""
        try:
            print("🔧 Essentia 서버 초기화 중...")

            # 메타데이터 로드
            self._load_metadata()

            # 모델 로드
            self._load_models()

            # 태그 카테고리화
            self._categorize_mood_tags()

            self.is_ready = True
            print("✅ Essentia 서버 초기화 완료!")

        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            traceback.print_exc()
            self.is_ready = False

    def _load_metadata(self):
        """JSON 메타데이터 로드"""
        try:
            # 장르 메타데이터
            genre_json_path = self.model_path / "discogs-effnet-bs64-1.json"
            with open(genre_json_path, 'r') as f:
                genre_metadata = json.load(f)
                self.genre_classes = genre_metadata.get('classes', [])
                print(f"✅ 장르 클래스 {len(self.genre_classes)}개 로드")

            # 분위기 메타데이터
            mood_json_path = self.model_path / "mtg_jamendo_moodtheme-discogs-effnet-1.json"
            with open(mood_json_path, 'r') as f:
                mood_metadata = json.load(f)
                self.mood_classes = mood_metadata.get('classes', [])
                print(f"✅ 분위기 클래스 {len(self.mood_classes)}개 로드")

        except Exception as e:
            raise Exception(f"메타데이터 로드 실패: {e}")

    def _load_models(self):
        """Essentia 모델들 로드"""
        try:
            # 모델 파일 경로
            discogs_model_path = self.model_path / "discogs-effnet-bs64-1.pb"
            mood_model_path = self.model_path / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"

            # 파일 존재 확인
            if not discogs_model_path.exists():
                raise FileNotFoundError(f"Discogs 모델 파일 없음: {discogs_model_path}")
            if not mood_model_path.exists():
                raise FileNotFoundError(f"Mood 모델 파일 없음: {mood_model_path}")

            # 임베딩 추출 모델
            self.embeddings_model = TensorflowPredictEffnetDiscogs(
                graphFilename=str(discogs_model_path),
                output="PartitionedCall:1"  # 임베딩 출력
            )
            print("✅ Discogs EfficientNet 임베딩 모델 로드")

            # 장르 분류 모델
            self.genre_model = TensorflowPredictEffnetDiscogs(
                graphFilename=str(discogs_model_path),
                output="PartitionedCall:0"  # 장르 예측 출력
            )
            print("✅ Discogs 장르 분류 모델 로드")

            # 분위기 분류 모델
            self.mood_model = TensorflowPredict2D(
                graphFilename=str(mood_model_path),
                output="model/Sigmoid"
            )
            print("✅ MTG Jamendo 분위기 모델 로드")

        except Exception as e:
            raise Exception(f"모델 로드 실패: {e}")

    def _categorize_mood_tags(self):
        """분위기 태그를 카테고리별로 분류"""

        # 분위기 관련 키워드들
        mood_keywords = {
            'calm', 'cool', 'dark', 'deep', 'dramatic', 'emotional', 'energetic',
            'epic', 'fast', 'fun', 'funny', 'groovy', 'happy', 'heavy', 'hopeful',
            'inspiring', 'meditative', 'melancholic', 'motivational', 'positive',
            'powerful', 'relaxing', 'romantic', 'sad', 'sexy', 'slow', 'soft',
            'upbeat', 'uplifting'
        }

        # 테마 관련 키워드들
        theme_keywords = {
            'action', 'adventure', 'ballad', 'children', 'christmas', 'dream',
            'film', 'game', 'holiday', 'love', 'movie', 'nature', 'party',
            'retro', 'space', 'sport', 'summer', 'travel'
        }

        # 기능 관련 키워드들
        function_keywords = {
            'advertising', 'background', 'commercial', 'corporate', 'documentary',
            'drama', 'soundscape', 'trailer'
        }

        # 카테고리별로 분류
        self.mood_tags = [tag for tag in self.mood_classes if tag in mood_keywords]
        self.theme_tags = [tag for tag in self.mood_classes if tag in theme_keywords]
        self.function_tags = [tag for tag in self.mood_classes if tag in function_keywords]

        print(f"📊 태그 카테고리화: 분위기({len(self.mood_tags)}) 테마({len(self.theme_tags)}) 기능({len(self.function_tags)})")

    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """오디오 파일 분석"""

        if not self.is_ready:
            raise Exception("서버가 초기화되지 않았습니다")

        try:
            print(f"🎵 오디오 분석 시작: {file_path}")

            # 오디오 로드
            audio = MonoLoader(filename=file_path, sampleRate=16000)()
            print(f"✅ 오디오 로드: {len(audio) / 16000:.1f}초")

            # numpy 배열로 변환 및 전처리
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            else:
                audio = audio.astype(np.float32)

            if not audio.flags['C_CONTIGUOUS']:
                audio = np.ascontiguousarray(audio)

            # 장르 분류
            genre_results = self._predict_genres(audio)

            # 분위기 분석
            mood_results = self._analyze_moods(audio)

            # 결과 구성
            result = {
                "status": "success",
                "audio_duration": len(audio) / 16000,
                "genres": {
                    "top_genres": genre_results[:5],
                    "all_genres": genre_results
                },
                "moods": mood_results["moods_info"],
                "all_activations": mood_results["all_activations"],
                "model_info": {
                    "using_essentia": True,
                    "genre_classes": len(self.genre_classes),
                    "mood_classes": len(self.mood_classes)
                },
                "timestamp": time.time()
            }

            print("✅ 분석 완료")
            return result

        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    def _predict_genres(self, audio: np.ndarray) -> list:
        """장르 예측"""
        try:
            # essentia 배열로 변환
            audio_essentia = essentia.array(audio)

            # 장르 예측
            predictions = self.genre_model(audio_essentia)

            # 패치별 예측을 평균내기
            if len(predictions) > 1:
                prediction_avs = []
                for i in range(len(predictions[0])):
                    vals = [predictions[j][i] for j in range(len(predictions))]
                    prediction_avs.append(sum(vals) / len(vals))
            else:
                prediction_avs = predictions[0]

            # 상위 10개 장르 추출
            top_indices = np.argsort(prediction_avs)[-10:][::-1]

            genre_results = []
            for idx in top_indices:
                if idx < len(self.genre_classes):
                    genre_results.append({
                        'genre': self.genre_classes[idx],
                        'score': float(prediction_avs[idx]),
                        'index': int(idx)
                    })

            return genre_results

        except Exception as e:
            print(f"❌ 장르 예측 실패: {e}")
            return []

    def _analyze_moods(self, audio: np.ndarray) -> Dict[str, Any]:
        """분위기 분석"""
        try:
            # essentia 배열로 변환
            audio_essentia = essentia.array(audio)

            # 임베딩 추출
            embeddings = self.embeddings_model(audio_essentia)

            # 분위기 분류
            activations = self.mood_model(embeddings)

            # 패치별 예측을 평균내기
            activation_avs = []
            for i in range(len(activations[0])):
                vals = [activations[j][i] for j in range(len(activations))]
                activation_avs.append(sum(vals) / len(vals))

            # 딕셔너리로 변환
            activations_dict = {}
            for ind, tag in enumerate(self.mood_classes):
                if ind < len(activation_avs):
                    activations_dict[tag] = float(activation_avs[ind])
                else:
                    activations_dict[tag] = 0.0

            # IQR 기반 임계값 계산
            values = list(activations_dict.values())
            q1 = np.quantile(values, 0.25)
            q3 = np.quantile(values, 0.75)
            outlier_threshold = q3 + (1.5 * (q3 - q1))

            # 임계값 이상의 태그 선택 (melodic 제외)
            prominent_tags = [
                tag for tag, score in activations_dict.items()
                if (score >= outlier_threshold) and (tag != 'melodic')
            ]

            # 카테고리별 분류
            moods = [tag for tag in prominent_tags if tag in self.mood_tags]
            themes = [tag for tag in prominent_tags if tag in self.theme_tags]
            functions = [tag for tag in prominent_tags if tag in self.function_tags]

            # 상위 분위기/테마 (전체)
            top_moods = sorted(activations_dict.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                "all_activations": activations_dict,
                "moods_info": {
                    "prominent_moods": moods,
                    "prominent_themes": themes,
                    "prominent_functions": functions,
                    "top_all": top_moods,
                    "threshold": float(outlier_threshold)
                }
            }

        except Exception as e:
            print(f"❌ 분위기 분석 실패: {e}")
            return {
                "all_activations": {},
                "moods_info": {
                    "prominent_moods": [],
                    "prominent_themes": [],
                    "prominent_functions": [],
                    "top_all": [],
                    "threshold": 0.0
                }
            }


# FastAPI 앱 생성
app = FastAPI(
    title="음악 분류 API 서버",
    description="Essentia 기반 음악 장르 및 분위기 분석 서버",
    version="1.0.0"
)

# CORS 설정 (Colab에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 서버 인스턴스
music_server = None


@app.on_event("startup")
async def startup_event():
    """서버 시작시 모델 로드"""
    global music_server

    # 모델 파일 경로 설정 (현재 디렉토리의 models 폴더)
    model_path = "./dependencies"

    print(f"📁 모델 경로: {model_path}")

    # 모델 파일 존재 확인
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ 모델 디렉토리가 없습니다: {model_path}")
        print("다음 파일들을 ./dependencies/ 디렉토리에 준비하세요:")
        print("  - discogs-effnet-bs64-1.pb")
        print("  - mtg_jamendo_moodtheme-discogs-effnet-1.pb")
        print("  - discogs-effnet-bs64-1.json")
        print("  - mtg_jamendo_moodtheme-discogs-effnet-1.json")
        return

    music_server = EssentiaServer(model_path)

    if music_server.is_ready:
        print("🚀 음악 분류 서버 준비 완료!")
    else:
        print("❌ 서버 초기화 실패")


@app.get("/")
async def root():
    """기본 엔드포인트"""
    return {
        "message": "음악 분류 API 서버",
        "status": "running",
        "server_ready": music_server.is_ready if music_server else False,
        "endpoints": {
            "analyze": "/analyze-music",
            "health": "/health",
            "status": "/status"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "server_ready": music_server.is_ready if music_server else False,
        "essentia_available": ESSENTIA_AVAILABLE,
        "timestamp": time.time()
    }


@app.get("/status")
async def get_status():
    """서버 상태 상세 정보"""
    if not music_server:
        return {"error": "서버가 초기화되지 않았습니다"}

    return {
        "server_ready": music_server.is_ready,
        "essentia_available": ESSENTIA_AVAILABLE,
        "model_info": {
            "genre_classes": len(music_server.genre_classes),
            "mood_classes": len(music_server.mood_classes),
            "mood_tags": len(music_server.mood_tags),
            "theme_tags": len(music_server.theme_tags),
            "function_tags": len(music_server.function_tags)
        } if music_server.is_ready else None,
        "timestamp": time.time()
    }


@app.post("/analyze-music")
async def analyze_music(file: UploadFile = File(...)):
    """음악 파일 분석 API"""

    # 서버 상태 확인
    if not music_server or not music_server.is_ready:
        raise HTTPException(
            status_code=503,
            detail="서버가 준비되지 않았습니다. /health 엔드포인트를 확인하세요."
        )

    # 파일 형식 확인
    allowed_types = {
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
        "audio/mp4", "audio/m4a", "audio/flac", "audio/x-flac"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식: {file.content_type}. "
                   f"지원 형식: {', '.join(allowed_types)}"
        )

    # 임시 파일로 저장
    temp_dir = tempfile.mkdtemp()
    temp_file_path = None

    try:
        # 파일 확장자 추출
        file_extension = Path(file.filename).suffix.lower()
        if not file_extension:
            file_extension = ".mp3"  # 기본값

        # 임시 파일 생성
        temp_file_path = Path(temp_dir) / f"temp_audio{file_extension}"

        # 파일 저장
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📁 임시 파일 저장: {temp_file_path}")

        # 음악 분석
        result = music_server.analyze_audio_file(str(temp_file_path))

        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ API 처리 오류: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")

    finally:
        # 임시 파일 정리
        try:
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink()
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"⚠️ 임시 파일 정리 실패: {e}")


if __name__ == "__main__":
    print("🎵 음악 분류 FastAPI 서버 시작")
    print("=" * 50)
    print("📋 시작 전 체크리스트:")
    print("1. Essentia 설치: pip install essentia-tensorflow")
    print("2. 모델 파일들을 ./models/ 디렉토리에 배치")
    print("3. ngrok 설치 및 실행: ngrok http 8000")
    print("=" * 50)

    # 서버 실행
    uvicorn.run(
        app,
        host="0.0.0.0",  # 모든 인터페이스에서 접근 허용
        port=8000,
        reload=False,  # 프로덕션에서는 False
        log_level="info"
    )