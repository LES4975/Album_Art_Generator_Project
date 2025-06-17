# album_art_core.py
# 앨범 아트 생성기 핵심 로직 모듈

import os
import sys
import tempfile
import json
from pathlib import Path
import time

# Stable Diffusion 관련 imports
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io


class AlbumArtGenerator:
    """앨범 아트 생성기 메인 클래스"""

    def __init__(self, music_classifier_path=None, sd_model_id="runwayml/stable-diffusion-v1-5"):
        """
        초기화

        Args:
            music_classifier_path: 음악 분류기 모듈 경로 (None이면 기본 경로 사용)
            sd_model_id: Stable Diffusion 모델 ID
        """
        self.music_classifier = None
        self.sd_pipeline = None
        self.sd_model_id = sd_model_id
        self.music_classifier_path = music_classifier_path

        self.setup_models()

    def setup_models(self):
        """모델들 초기화"""
        print("🔧 모델 초기화 중...")

        # 음악 분류기 초기화
        self._setup_music_classifier()

        # Stable Diffusion 파이프라인 초기화
        self._setup_stable_diffusion()

    def _setup_music_classifier(self):
        """음악 분류기 초기화"""
        try:
            # 경로 설정
            if self.music_classifier_path:
                sys.path.append(self.music_classifier_path)
            else:
                # 기본 경로 (Google Drive)
                sys.path.append('/content/drive/MyDrive/album_art_generator_project')

            # 음악 분류기 임포트
            from music_classification_colab import ColabMusicClassifier

            self.music_classifier = ColabMusicClassifier()
            print("✅ 음악 분류기 초기화 완료")

        except ImportError as e:
            print(f"❌ 음악 분류기 로드 실패: {e}")
            print("Google Drive 마운트 및 파일 경로를 확인하세요")
            self.music_classifier = None
        except Exception as e:
            print(f"❌ 음악 분류기 초기화 실패: {e}")
            self.music_classifier = None

    def _setup_stable_diffusion(self):
        """Stable Diffusion 파이프라인 초기화"""
        try:
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                self.sd_model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )

            if torch.cuda.is_available():
                self.sd_pipeline = self.sd_pipeline.to("cuda")
                print("✅ Stable Diffusion GPU 로드 완료")
            else:
                print("✅ Stable Diffusion CPU 로드 완료 (속도 느림)")

        except Exception as e:
            print(f"❌ Stable Diffusion 초기화 실패: {e}")
            self.sd_pipeline = None

    def is_ready(self):
        """모델들이 준비되었는지 확인"""
        return self.music_classifier is not None and self.sd_pipeline is not None

    def get_status(self):
        """현재 상태 반환"""
        status = {
            "music_classifier": self.music_classifier is not None,
            "stable_diffusion": self.sd_pipeline is not None,
            "gpu_available": torch.cuda.is_available()
        }
        return status

    def analyze_music(self, audio_file):
        """음악 파일 분석"""
        if self.music_classifier is None:
            return {"error": "음악 분류기가 초기화되지 않았습니다"}

        if audio_file is None:
            return {"error": "음악 파일이 업로드되지 않았습니다"}

        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file)
                tmp_path = tmp_file.name

            # 음악 분석
            result = self.music_classifier.classify_music(tmp_path)

            # 임시 파일 삭제
            os.unlink(tmp_path)

            return result

        except Exception as e:
            return {"error": f"음악 분석 실패: {str(e)}"}

    def create_prompt_from_music_analysis(self, music_result, music_title="Unknown"):
        """음악 분석 결과를 바탕으로 프롬프트 생성"""

        if "error" in music_result:
            return f"Album cover for {music_title}, artistic and creative design", "text, letters, words, watermark, signature, blurry, low quality, ugly"

        # 상위 장르 추출
        top_genres = music_result.get('genres', {}).get('top_genres', [])
        genre_text = top_genres[0]['genre'] if top_genres else "music"

        # 주요 분위기 추출
        moods = music_result.get('moods', {})
        prominent_moods = moods.get('prominent_moods', [])
        prominent_themes = moods.get('prominent_themes', [])

        # 분위기 텍스트 생성
        mood_text = ""
        if prominent_moods:
            mood_text = ", ".join(prominent_moods[:3])  # 상위 3개만
        elif moods.get('top_all'):
            # 임계값을 넘지 않아도 상위 분위기 사용
            top_moods = [item[0] for item in moods['top_all'][:3]]
            mood_text = ", ".join(top_moods)

        # 테마 텍스트 생성
        theme_text = ""
        if prominent_themes:
            theme_text = ", ".join(prominent_themes[:2])  # 상위 2개만

        # 프롬프트 조합
        prompt_parts = [f"Album cover for '{music_title}'"]

        if genre_text:
            prompt_parts.append(f"{genre_text} genre")

        if mood_text:
            prompt_parts.append(f"{mood_text} mood")

        if theme_text:
            prompt_parts.append(f"{theme_text} theme")

        # 기본 스타일 추가
        prompt_parts.extend([
            "artistic album cover design",
            "professional music artwork",
            "high quality",
            "detailed illustration"
        ])

        final_prompt = ", ".join(prompt_parts)

        # 네거티브 프롬프트
        negative_prompt = "text, letters, words, watermark, signature, blurry, low quality, ugly"

        return final_prompt, negative_prompt

    def generate_album_art(self, prompt, negative_prompt="", **generation_kwargs):
        """
        Stable Diffusion으로 앨범 아트 생성

        Args:
            prompt: 생성 프롬프트
            negative_prompt: 네거티브 프롬프트
            **generation_kwargs: 추가 생성 파라미터
        """
        if self.sd_pipeline is None:
            return None, "Stable Diffusion 모델이 로드되지 않았습니다"

        try:
            print(f"🎨 이미지 생성 시작...")
            print(f"프롬프트: {prompt}")

            # 기본 생성 파라미터
            default_params = {
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512
            }

            # 사용자 파라미터로 덮어쓰기
            default_params.update(generation_kwargs)

            # 이미지 생성
            image = self.sd_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                **default_params
            ).images[0]

            print("✅ 이미지 생성 완료")
            return image, None

        except Exception as e:
            error_msg = f"이미지 생성 실패: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg

    def process_music_to_art(self, audio_file, **generation_kwargs):
        """
        전체 파이프라인: 음악 → 앨범 아트

        Args:
            audio_file: 음악 파일 경로
            **generation_kwargs: 이미지 생성 파라미터
        """

        # 입력 검증
        if audio_file is None:
            return None, "❌ 음악 파일을 업로드해주세요", "", ""

        # 파일명에서 제목 추출 (확장자 제거)
        music_title = Path(audio_file).stem
        print(f"📁 추출된 곡 제목: {music_title}")

        # 1. 음악 분석
        print("🔄 1단계: 음악 분석 중...")
        with open(audio_file, 'rb') as f:
            audio_data = f.read()

        music_result = self.analyze_music(audio_data)

        if "error" in music_result:
            return None, f"❌ 음악 분석 실패: {music_result['error']}", "", ""

        # 2. 분석 결과 텍스트 생성
        analysis_text = self.format_analysis_result(music_result)

        # 3. 프롬프트 생성
        print("🔄 2단계: 프롬프트 생성 중...")
        prompt, negative_prompt = self.create_prompt_from_music_analysis(music_result, music_title)

        # 4. 이미지 생성
        print("🔄 3단계: 앨범 아트 생성 중...")
        generated_image, error = self.generate_album_art(prompt, negative_prompt, **generation_kwargs)

        if error:
            return None, f"❌ {error}", analysis_text, prompt

        return generated_image, "✅ 앨범 아트 생성 완료!", analysis_text, prompt

    def format_analysis_result(self, result):
        """분석 결과를 보기 좋게 포맷팅"""
        if "error" in result:
            return f"오류: {result['error']}"

        text_parts = []

        # 기본 정보
        duration = result.get('audio_duration', 0)
        text_parts.append(f"⏱️ 길이: {duration:.1f}초\n")

        # 장르 정보
        genres = result.get('genres', {}).get('top_genres', [])
        if genres:
            text_parts.append("🎼 상위 장르:")
            for i, genre_info in enumerate(genres[:3], 1):
                text_parts.append(f"  {i}. {genre_info['genre']}: {genre_info['score']:.3f}")
            text_parts.append("")

        # 분위기 정보
        moods = result.get('moods', {})
        prominent_moods = moods.get('prominent_moods', [])
        prominent_themes = moods.get('prominent_themes', [])

        if prominent_moods:
            text_parts.append("🎭 주요 분위기:")
            for mood in prominent_moods:
                score = result['all_activations'].get(mood, 0)
                text_parts.append(f"  • {mood}: {score:.3f}")
            text_parts.append("")

        if prominent_themes:
            text_parts.append("🎨 주요 테마:")
            for theme in prominent_themes:
                score = result['all_activations'].get(theme, 0)
                text_parts.append(f"  • {theme}: {score:.3f}")
            text_parts.append("")

        # 상위 분위기 (전체)
        top_all = moods.get('top_all', [])
        if top_all:
            text_parts.append("📊 상위 분위기 (전체):")
            for i, (mood, score) in enumerate(top_all[:5], 1):
                text_parts.append(f"  {i}. {mood}: {score:.3f}")

        return "\n".join(text_parts)


# 편의 함수들
def create_album_art_generator(music_classifier_path=None, sd_model_id="runwayml/stable-diffusion-v1-5"):
    """앨범 아트 생성기 인스턴스 생성"""
    return AlbumArtGenerator(music_classifier_path, sd_model_id)


def extract_music_title(file_path):
    """파일 경로에서 음악 제목 추출"""
    return Path(file_path).stem


def save_album_art(image, output_path, music_title=None):
    """앨범 아트 이미지 저장"""
    if music_title:
        filename = f"album_art_{music_title}_{int(time.time())}.png"
        save_path = Path(output_path) / filename
    else:
        save_path = Path(output_path)

    image.save(save_path)
    return str(save_path)