# music_classification_colab.py (간소화 버전)
# 순수 음악 분류 로직만 포함!

import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Essentia 라이브러리 import
from essentia.standard import (
    AudioLoader,
    MonoLoader,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D
)
import essentia

class ColabMusicClassifier:
    """Google Colab용 음악 분류기 - Essentia 네이티브 사용"""

    def __init__(self):
        """분류기 초기화 (모델 파일들이 이미 준비되어 있다고 가정)"""

        # 메타데이터 로드
        self.genre_classes = []
        self.mood_classes = []
        self._load_metadata()

        # Essentia 모델 초기화
        self._initialize_essentia_models()

        # 분위기 태그 카테고리화
        self._categorize_mood_tags()

    def _load_metadata(self):
        """JSON 메타데이터 로드"""
        try:
            # 장르 메타데이터
            with open("discogs-effnet-bs64-1.json", 'r') as f:
                genre_metadata = json.load(f)
                self.genre_classes = genre_metadata.get('classes', [])
                print(f"✅ 장르 클래스 {len(self.genre_classes)}개 로드 완료")

            # 분위기 메타데이터
            with open("mtg_jamendo_moodtheme-discogs-effnet-1.json", 'r') as f:
                mood_metadata = json.load(f)
                self.mood_classes = mood_metadata.get('classes', [])
                print(f"✅ 분위기 클래스 {len(self.mood_classes)}개 로드 완료")

        except Exception as e:
            print(f"❌ 메타데이터 로드 실패: {e}")

    def _initialize_essentia_models(self):
        """Essentia 모델 초기화"""
        try:
            # 임베딩 추출 모델
            self.embeddings_model = TensorflowPredictEffnetDiscogs(
                graphFilename="discogs-effnet-bs64-1.pb",
                output="PartitionedCall:1"  # 임베딩 출력
            )
            print("✅ Discogs EfficientNet 모델 로드 완료")

            # 분위기 분류 모델
            self.mood_model = TensorflowPredict2D(
                graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb",
                output="model/Sigmoid"
            )
            print("✅ MTG Jamendo 분위기 모델 로드 완료")

            # 장르 분류용 모델 (예측 출력)
            self.genre_model = TensorflowPredictEffnetDiscogs(
                graphFilename="discogs-effnet-bs64-1.pb",
                output="PartitionedCall:0"  # 장르 예측 출력
            )
            print("✅ 장르 분류 모델 로드 완료")

        except Exception as e:
            print(f"❌ Essentia 모델 초기화 실패: {e}")

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

        print(f"📊 분위기 태그 카테고리화 완료:")
        print(f"   • 분위기: {len(self.mood_tags)}개")
        print(f"   • 테마: {len(self.theme_tags)}개")
        print(f"   • 기능: {len(self.function_tags)}개")

    def get_mood_activations_dict(self, audio):
        """
        분위기 활성화 계산 (numpy 호환성 수정)
        """
        try:
            # numpy 배열을 essentia 배열로 변환
            if isinstance(audio, np.ndarray):
                audio_essentia = essentia.array(audio.astype(np.float32))
            else:
                audio_essentia = audio

            print(f"🔍 분위기 분석 - 오디오 타입: {type(audio_essentia)}")

            # Essentia로 임베딩 추출
            embeddings = self.embeddings_model(audio_essentia)
            print(f"🔍 임베딩 추출 완료: {type(embeddings)}")

            # 분위기 분류
            activations = self.mood_model(embeddings)
            print(f"🔍 분위기 예측 완료: {type(activations)}")

            # 패치별 예측을 평균내기
            activation_avs = []
            for i in range(len(activations[0])):
                vals = [activations[j][i] for j in range(len(activations))]
                activation_avs.append(sum(vals) / len(vals))

            # 딕셔너리로 변환
            activations_dict = {}
            for ind, tag in enumerate(self.mood_classes):
                if ind < len(activation_avs):
                    activations_dict[tag] = activation_avs[ind]
                else:
                    activations_dict[tag] = 0.0

            return activations_dict

        except Exception as e:
            print(f"❌ 분위기 활성화 계산 실패: {e}")
            print(f"   오디오 타입: {type(audio)}")
            if hasattr(audio, 'shape'):
                print(f"   오디오 형태: {audio.shape}")
            return {}

    def get_genre_predictions(self, audio):
        """장르 예측 (numpy 호환성 수정)"""
        try:
            # numpy 배열을 essentia 배열로 변환
            if isinstance(audio, np.ndarray):
                audio_essentia = essentia.array(audio.astype(np.float32))
            else:
                audio_essentia = audio

            print(f"🔍 오디오 타입: {type(audio_essentia)}, 형태: {audio_essentia.shape}")

            # Essentia로 장르 예측
            predictions = self.genre_model(audio_essentia)

            print(f"🔍 예측 결과 타입: {type(predictions)}")

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
            print(f"   오디오 타입: {type(audio)}")
            if hasattr(audio, 'shape'):
                print(f"   오디오 형태: {audio.shape}")
            if hasattr(audio, 'dtype'):
                print(f"   오디오 dtype: {audio.dtype}")
            return []

    def get_moods(self, audio):
        """
        분위기 추출
        """
        # 분위기 활성화 계산
        mood_activations_dict = self.get_mood_activations_dict(essentia.array(audio))

        # IQR 기반 임계값 계산
        values = list(mood_activations_dict.values())
        q1 = np.quantile(values, 0.25)
        q3 = np.quantile(values, 0.75)
        outlier_threshold = q3 + (1.5 * (q3 - q1))

        # 임계값 이상의 태그 선택 (melodic 제외)
        prominent_tags = [
            tag for tag, score in mood_activations_dict.items()
            if (score >= outlier_threshold) and (tag != 'melodic')
        ]

        # 카테고리별 분류
        moods = [tag for tag in prominent_tags if tag in self.mood_tags]
        themes = [tag for tag in prominent_tags if tag in self.theme_tags]
        functions = [tag for tag in prominent_tags if tag in self.function_tags]

        return moods, themes, functions, mood_activations_dict, outlier_threshold

    def classify_music(self, audio_file):
        """음악 파일 종합 분류 (numpy 호환성 개선)"""
        print(f"🎵 음악 분석 시작: {audio_file}")
        print("=" * 60)

        try:
            # 오디오 로드 (Essentia 방식)
            print("🔄 오디오 로드 중...")
            if isinstance(audio_file, str):
                # 파일 경로인 경우
                audio = MonoLoader(filename=audio_file, sampleRate=16000)()
            else:
                # 이미 로드된 오디오인 경우
                audio = audio_file

            print(f"✅ 오디오 로드 완료: {len(audio)/16000:.1f}초")
            print(f"🔍 오디오 정보: 타입={type(audio)}, 형태={audio.shape}, dtype={audio.dtype}")

            # numpy 배열로 확실히 변환 및 타입 체크
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            else:
                audio = audio.astype(np.float32)

            # 메모리 레이아웃 확인 및 수정
            if not audio.flags['C_CONTIGUOUS']:
                audio = np.ascontiguousarray(audio)

            print(f"🔍 전처리 후 오디오: 타입={type(audio)}, 형태={audio.shape}, dtype={audio.dtype}")

            # 장르 분류
            print("🔄 장르 분류 중...")
            genre_results = self.get_genre_predictions(audio)
            print(f"✅ 장르 분류 완료: 상위 {len(genre_results)}개")

            # 분위기 분류
            print("🔄 분위기 분석 중...")
            moods, themes, functions, all_activations, threshold = self.get_moods(audio)
            print(f"✅ 분위기 분석 완료")

            # 상위 분위기/테마 (전체)
            top_moods = sorted(all_activations.items(), key=lambda x: x[1], reverse=True)[:10]

            # 결과 정리
            result = {
                "audio_duration": len(audio) / 16000,
                "genres": {
                    "top_genres": genre_results[:5],
                    "all_genres": genre_results
                },
                "moods": {
                    "prominent_moods": moods,
                    "prominent_themes": themes,
                    "prominent_functions": functions,
                    "top_all": top_moods,
                    "threshold": threshold
                },
                "all_activations": all_activations,
                "model_info": {
                    "using_essentia": True,
                    "genre_classes": len(self.genre_classes),
                    "mood_classes": len(self.mood_classes)
                }
            }

            return result

        except Exception as e:
            print(f"❌ 전체 분석 실패: {e}")
            return {"error": f"분석 실패: {e}"}

# 결과 출력 함수
def print_results(result):
    """분류 결과를 보기 좋게 출력"""

    if "error" in result:
        print(f"❌ 오류: {result['error']}")
        return

    print("\n" + "="*60)
    print("🎵 음악 분류 결과 (Essentia 네이티브)")
    print("="*60)

    print(f"⏱️  길이: {result['audio_duration']:.1f}초")

    print(f"\n🎼 상위 장르:")
    for i, genre_info in enumerate(result['genres']['top_genres'], 1):
        print(f"  {i}. {genre_info['genre']}: {genre_info['score']:.4f}")

    print(f"\n🎭 주요 분위기 (임계값: {result['moods']['threshold']:.4f}):")
    if result['moods']['prominent_moods']:
        for mood in result['moods']['prominent_moods']:
            score = result['all_activations'][mood]
            print(f"  • {mood}: {score:.4f}")
    else:
        print("  (임계값을 넘는 분위기 없음)")

    print(f"\n🎨 주요 테마:")
    if result['moods']['prominent_themes']:
        for theme in result['moods']['prominent_themes']:
            score = result['all_activations'][theme]
            print(f"  • {theme}: {score:.4f}")
    else:
        print("  (임계값을 넘는 테마 없음)")

    print(f"\n⚙️  주요 기능:")
    if result['moods']['prominent_functions']:
        for function in result['moods']['prominent_functions']:
            score = result['all_activations'][function]
            print(f"  • {function}: {score:.4f}")
    else:
        print("  (임계값을 넘는 기능 없음!)")

    print(f"\n📊 상위 분위기/테마 (전체):")
    for i, (mood, score) in enumerate(result['moods']['top_all'], 1):
        print(f"  {i}. {mood}: {score:.4f}")

    print(f"\n🤖 모델 정보:")
    print(f"  • Essentia 네이티브 사용: {result['model_info']['using_essentia']}")
    print(f"  • 장르 클래스: {result['model_info']['genre_classes']}개")
    print(f"  • 분위기 클래스: {result['model_info']['mood_classes']}개")