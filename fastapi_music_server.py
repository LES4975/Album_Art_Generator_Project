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
        서버 초기화

        Args:
            model_path: 사전학습된 모델 파일들이 있는 디렉토리 경로
        """
        self.model_path = Path(model_path)

        # 단계 1: 클래스 정보 저장을 위한 변수 초기화
        self.genre_classes = []  # 400개 장르 클래스명
        self.mood_classes = []  # 87개 분위기/테마 클래스명

        # 단계 2: AI 모델 객체들을 위한 변수 초기화
        self.embeddings_model = None  # 음악 특징 추출용 모델
        self.mood_model = None  # 분위기 분석용 모델
        self.genre_model = None  # 장르 분류용 모델

        # 단계 3: 분위기 태그 카테고리별 분류를 위한 리스트
        self.mood_tags = []  # 감정/분위기 관련 태그
        self.theme_tags = []  # 테마/장면 관련 태그
        self.function_tags = []  # 기능/용도 관련 태그

        # 단계 4: 서버 상태 플래그
        self.is_ready = False

        # 단계 5: Essentia가 사용 가능한 경우에만 초기화 진행
        if ESSENTIA_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """모델 및 메타데이터 초기화"""
        try:
            print("🔧 Essentia 서버 초기화 중...")

            # 단계 1: JSON 파일에서 클래스 정보 로드
            self._load_metadata()

            # 단계 2: 사전학습된 TensorFlow 모델들 로드
            self._load_models()

            # 단계 3: 분위기 태그를 용도별로 카테고리 분류
            self._categorize_mood_tags()

            # 단계 4: 서버 준비 완료 상태로 설정
            self.is_ready = True
            print("✅ Essentia 서버 초기화 완료!")

        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            traceback.print_exc()
            self.is_ready = False

    def _load_metadata(self):
        """JSON 메타데이터 로드"""
        try:
            # 단계 1: 장르 분류용 메타데이터 로드
            genre_json_path = self.model_path / "discogs-effnet-bs64-1.json"
            with open(genre_json_path, 'r') as f:
                genre_metadata = json.load(f)
                self.genre_classes = genre_metadata.get('classes', [])
                print(f"✅ 장르 클래스 {len(self.genre_classes)}개 로드")

            # 단계 2: 분위기 분석용 메타데이터 로드
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
            # 단계 1: 모델 파일 경로 설정
            discogs_model_path = self.model_path / "discogs-effnet-bs64-1.pb"
            mood_model_path = self.model_path / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"

            # 단계 2: 모델 파일 존재 여부 확인
            if not discogs_model_path.exists():
                raise FileNotFoundError(f"Discogs 모델 파일 없음: {discogs_model_path}")
            if not mood_model_path.exists():
                raise FileNotFoundError(f"Mood 모델 파일 없음: {mood_model_path}")

            # 단계 3: EfficientNet 기반 임베딩 추출 모델 로드
            self.embeddings_model = TensorflowPredictEffnetDiscogs(
                graphFilename=str(discogs_model_path),
                output="PartitionedCall:1"  # 음악 특징 벡터 출력
            )
            print("✅ Discogs EfficientNet 임베딩 모델 로드")

            # 단계 4: 장르 분류 모델 로드 (같은 EfficientNet, 다른 출력)
            self.genre_model = TensorflowPredictEffnetDiscogs(
                graphFilename=str(discogs_model_path),
                output="PartitionedCall:0"  # 장르 확률 출력
            )
            print("✅ Discogs 장르 분류 모델 로드")

            # 단계 5: 분위기 분석 모델 로드 (임베딩을 입력으로 받음)
            self.mood_model = TensorflowPredict2D(
                graphFilename=str(mood_model_path),
                output="model/Sigmoid"  # 분위기 확률 출력
            )
            print("✅ MTG Jamendo 분위기 모델 로드")

        except Exception as e:
            raise Exception(f"모델 로드 실패: {e}")

    def _categorize_mood_tags(self):
        """분위기 태그를 카테고리별로 분류"""

        # 단계 1: 감정/분위기 관련 키워드 정의
        mood_keywords = {
            'calm', 'cool', 'dark', 'deep', 'dramatic', 'emotional', 'energetic',
            'epic', 'fast', 'fun', 'funny', 'groovy', 'happy', 'heavy', 'hopeful',
            'inspiring', 'meditative', 'melancholic', 'motivational', 'positive',
            'powerful', 'relaxing', 'romantic', 'sad', 'sexy', 'slow', 'soft',
            'upbeat', 'uplifting'
        }

        # 단계 2: 테마/장면 관련 키워드 정의
        theme_keywords = {
            'action', 'adventure', 'ballad', 'children', 'christmas', 'dream',
            'film', 'game', 'holiday', 'love', 'movie', 'nature', 'party',
            'retro', 'space', 'sport', 'summer', 'travel'
        }

        # 단계 3: 기능/용도 관련 키워드 정의
        function_keywords = {
            'advertising', 'background', 'commercial', 'corporate', 'documentary',
            'drama', 'soundscape', 'trailer'
        }

        # 단계 4: 전체 분위기 클래스를 3개 카테고리로 분류
        self.mood_tags = [tag for tag in self.mood_classes if tag in mood_keywords]
        self.theme_tags = [tag for tag in self.mood_classes if tag in theme_keywords]
        self.function_tags = [tag for tag in self.mood_classes if tag in function_keywords]

        print(f"📊 태그 카테고리화: 분위기({len(self.mood_tags)}) 테마({len(self.theme_tags)}) 기능({len(self.function_tags)})")

    def analyze_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        오디오 파일 분석 메인 함수
        입력: 오디오 파일 경로 → 출력: 장르 + 분위기 분석 결과
        """

        # 단계 1: 서버 준비 상태 확인
        if not self.is_ready:
            raise Exception("서버가 초기화되지 않았습니다")

        try:
            print(f"🎵 오디오 분석 시작: {file_path}")

            # 단계 2: 오디오 파일을 16kHz 모노로 로드
            audio = MonoLoader(filename=file_path, sampleRate=16000)()
            print(f"✅ 오디오 로드: {len(audio) / 16000:.1f}초")

            # 단계 3: numpy 배열로 변환 및 메모리 최적화
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            else:
                audio = audio.astype(np.float32)

            # C-연속 메모리 레이아웃으로 변환 (성능 최적화)
            if not audio.flags['C_CONTIGUOUS']:
                audio = np.ascontiguousarray(audio)

            # 단계 4: AI 모델을 사용한 장르 분류 수행
            genre_results = self._predict_genres(audio)

            # 단계 5: AI 모델을 사용한 분위기 분석 수행
            mood_results = self._analyze_moods(audio)

            # 단계 6: 분석 결과를 구조화하여 JSON 형태로 구성
            result = {
                "status": "success",
                "audio_duration": len(audio) / 16000,
                "genres": {
                    "top_genres": genre_results[:5],  # 상위 5개 장르
                    "all_genres": genre_results  # 전체 장르 결과
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
        """
        장르 예측 수행
        입력: 전처리된 오디오 → 출력: 상위 장르 리스트
        """
        try:
            # 단계 1: numpy 배열을 Essentia 배열로 변환
            audio_essentia = essentia.array(audio)

            # 단계 2: EfficientNet 모델로 장르 확률 예측
            predictions = self.genre_model(audio_essentia)

            # 단계 3: 여러 패치의 예측 결과를 평균내어 최종 점수 계산
            if len(predictions) > 1:
                prediction_avs = []
                for i in range(len(predictions[0])):
                    vals = [predictions[j][i] for j in range(len(predictions))]
                    prediction_avs.append(sum(vals) / len(vals))
            else:
                prediction_avs = predictions[0]

            # 단계 4: 확률이 높은 상위 10개 장르 인덱스 추출
            top_indices = np.argsort(prediction_avs)[-10:][::-1]

            # 단계 5: 장르명과 점수를 딕셔너리 형태로 구성
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
        """
        분위기 분석 수행
        입력: 전처리된 오디오 → 출력: 분위기/테마 분석 결과
        """
        try:
            # 단계 1: numpy 배열을 Essentia 배열로 변환
            audio_essentia = essentia.array(audio)

            # 단계 2: EfficientNet으로 음악 특징 임베딩 추출
            embeddings = self.embeddings_model(audio_essentia)

            # 단계 3: 임베딩을 분위기 분류 모델에 입력하여 분위기 확률 계산
            activations = self.mood_model(embeddings)

            # 단계 4: 여러 패치의 예측을 평균내어 최종 분위기 점수 계산
            activation_avs = []
            for i in range(len(activations[0])):
                vals = [activations[j][i] for j in range(len(activations))]
                activation_avs.append(sum(vals) / len(vals))

            # 단계 5: 분위기 태그와 점수를 딕셔너리로 매핑
            activations_dict = {}
            for ind, tag in enumerate(self.mood_classes):
                if ind < len(activation_avs):
                    activations_dict[tag] = float(activation_avs[ind])
                else:
                    activations_dict[tag] = 0.0

            # 단계 6: IQR 방법으로 통계적 임계값 계산
            values = list(activations_dict.values())
            q1 = np.quantile(values, 0.25)  # 1사분위수
            q3 = np.quantile(values, 0.75)  # 3사분위수
            outlier_threshold = q3 + (1.5 * (q3 - q1))  # 이상치 기준

            # 단계 7: 임계값 이상의 의미 있는 태그만 선별 (melodic 제외)
            prominent_tags = [
                tag for tag, score in activations_dict.items()
                if (score >= outlier_threshold) and (tag != 'melodic')
            ]

            # 단계 8: 선별된 태그를 카테고리별로 분류
            moods = [tag for tag in prominent_tags if tag in self.mood_tags]
            themes = [tag for tag in prominent_tags if tag in self.theme_tags]
            functions = [tag for tag in prominent_tags if tag in self.function_tags]

            # 단계 9: 전체 분위기를 점수 순으로 정렬 (상위 10개)
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


# FastAPI 앱 생성 및 설정
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
    """
    서버 시작시 모델 로드
    FastAPI 서버가 시작될 때 자동으로 실행되는 이벤트 핸들러
    """
    global music_server

    # 단계 1: 모델 파일 경로 설정 (현재 디렉토리의 dependencies 폴더)
    model_path = "./dependencies"
    print(f"📁 모델 경로: {model_path}")

    # 단계 2: 필수 모델 파일들의 존재 여부 확인
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ 모델 디렉토리가 없습니다: {model_path}")
        print("다음 파일들을 ./dependencies/ 디렉토리에 준비하세요:")
        print("  - discogs-effnet-bs64-1.pb")
        print("  - mtg_jamendo_moodtheme-discogs-effnet-1.pb")
        print("  - discogs-effnet-bs64-1.json")
        print("  - mtg_jamendo_moodtheme-discogs-effnet-1.json")
        return

    # 단계 3: Essentia 서버 인스턴스 생성 및 초기화
    music_server = EssentiaServer(model_path)

    # 단계 4: 서버 준비 상태 확인 및 로그 출력
    if music_server.is_ready:
        print("🚀 음악 분류 서버 준비 완료!")
    else:
        print("❌ 서버 초기화 실패")


@app.get("/")
async def root():
    """기본 엔드포인트 - 서버 정보 제공"""
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
    """헬스 체크 - 서버 상태 간단 확인"""
    return {
        "status": "healthy",
        "server_ready": music_server.is_ready if music_server else False,
        "essentia_available": ESSENTIA_AVAILABLE,
        "timestamp": time.time()
    }


@app.get("/status")
async def get_status():
    """서버 상태 상세 정보 - 모델 로딩 상태 및 클래스 수 확인"""
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
    """
    음악 파일 분석 API 메인 엔드포인트
    입력: 업로드된 음악 파일 → 출력: 장르 + 분위기 분석 결과 JSON
    """

    # 단계 1: 서버 준비 상태 확인
    if not music_server or not music_server.is_ready:
        raise HTTPException(
            status_code=503,
            detail="서버가 준비되지 않았습니다. /health 엔드포인트를 확인하세요."
        )

    # 단계 2: 업로드된 파일의 형식 검증
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

    # 단계 3: 임시 디렉토리 및 파일 생성
    temp_dir = tempfile.mkdtemp()
    temp_file_path = None

    try:
        # 단계 4: 파일 확장자 추출 (기본값: .mp3)
        file_extension = Path(file.filename).suffix.lower()
        if not file_extension:
            file_extension = ".mp3"

        # 단계 5: 업로드된 파일을 임시 파일로 저장
        temp_file_path = Path(temp_dir) / f"temp_audio{file_extension}"

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"📁 임시 파일 저장: {temp_file_path}")

        # 단계 6: Essentia 서버를 통한 음악 분석 수행
        result = music_server.analyze_audio_file(str(temp_file_path))

        # 단계 7: 분석 결과를 JSON 응답으로 반환
        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ API 처리 오류: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")

    finally:
        # 단계 8: 임시 파일 및 디렉토리 정리 (메모리 누수 방지)
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
    print("2. 모델 파일들을 ./dependencies/ 디렉토리에 배치")
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