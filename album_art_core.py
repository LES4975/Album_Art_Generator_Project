# album_art_core.py
# SDXL Lightning 기반 앨범 아트 생성기 (업데이트 버전)

import os
import sys
import tempfile
import json
from pathlib import Path
import time

# SDXL Lightning 관련 imports
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import io


class AlbumArtGenerator:
    """앨범 아트 생성기 메인 클래스 (SDXL Lightning 기반)"""

    def __init__(self, music_classifier_path=None,
                 base_model="stabilityai/stable-diffusion-xl-base-1.0",
                 lightning_repo="ByteDance/SDXL-Lightning",
                 lightning_steps=4):
        """
        초기화

        Args:
            music_classifier_path: 음악 분류기 모듈 경로 (None이면 기본 경로 사용)
            base_model: 기본 SDXL 모델 ID
            lightning_repo: SDXL Lightning LoRA 리포지토리
            lightning_steps: Lightning 스텝 수 (2, 4, 8 중 선택)
        """
        self.music_classifier = None
        self.sd_pipeline = None
        self.base_model = base_model
        self.lightning_repo = lightning_repo
        self.lightning_steps = lightning_steps
        self.music_classifier_path = music_classifier_path

        # Lightning 모델 파일명 매핑
        self.lightning_files = {
            2: "sdxl_lightning_2step_lora.safetensors",
            4: "sdxl_lightning_4step_lora.safetensors",
            8: "sdxl_lightning_8step_lora.safetensors"
        }

        self.setup_models()

    def setup_models(self):
        """모델들 초기화"""
        print("🔧 모델 초기화 중...")

        # 음악 분류기 초기화
        self._setup_music_classifier()

        # SDXL Lightning 파이프라인 초기화
        self._setup_sdxl_lightning()

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

    def _setup_sdxl_lightning(self):
        """SDXL Lightning 파이프라인 초기화"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔧 SDXL Lightning 로딩 중... (디바이스: {device})")

            # 기본 SDXL 파이프라인 로드
            print(f"📦 기본 SDXL 모델 로딩: {self.base_model}")
            self.sd_pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
                use_safetensors=True
            )

            # Lightning LoRA 다운로드 및 적용
            lightning_file = self.lightning_files.get(self.lightning_steps)
            if not lightning_file:
                raise ValueError(f"지원하지 않는 스텝 수: {self.lightning_steps}")

            print(f"📦 Lightning {self.lightning_steps}스텝 LoRA 로딩...")
            lightning_lora_path = hf_hub_download(
                repo_id=self.lightning_repo,
                filename=lightning_file
            )

            # LoRA 어댑터 로드
            self.sd_pipeline.load_lora_weights(lightning_lora_path)

            # Lightning용 스케줄러 설정
            self.sd_pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.sd_pipeline.scheduler.config,
                timestep_spacing="trailing"
            )

            # GPU 최적화
            if device == "cuda":
                self.sd_pipeline = self.sd_pipeline.to("cuda")

                # T4 GPU 메모리 최적화
                self.sd_pipeline.enable_attention_slicing()
                self.sd_pipeline.enable_model_cpu_offload()
                self.sd_pipeline.enable_vae_slicing()

                print("✅ SDXL Lightning GPU 로드 완료 (최적화)")
            else:
                print("✅ SDXL Lightning CPU 로드 완료 (속도 느림)")

        except Exception as e:
            print(f"❌ SDXL Lightning 초기화 실패: {e}")
            self.sd_pipeline = None

    def is_ready(self):
        """모델들이 준비되었는지 확인"""
        return self.music_classifier is not None and self.sd_pipeline is not None

    def get_status(self):
        """현재 상태 반환"""
        status = {
            "music_classifier": self.music_classifier is not None,
            "sdxl_lightning": self.sd_pipeline is not None,
            "lightning_steps": self.lightning_steps,
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
        """음악 분석 결과를 바탕으로 프롬프트 생성 (SDXL Lightning 최적화)"""

        if "error" in music_result:
            return (f"Album cover for {music_title}, artistic and creative design, professional, high quality",
                    "text, letters, words, watermark, signature, blurry, low quality, ugly")

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

        # 프롬프트 조합 (SDXL Lightning에 최적화)
        prompt_parts = [f"Album cover for '{music_title}'"]

        if genre_text:
            prompt_parts.append(f"{genre_text} genre")

        if mood_text:
            prompt_parts.append(f"{mood_text} mood")

        if theme_text:
            prompt_parts.append(f"{theme_text} theme")

        # SDXL Lightning에 적합한 스타일 키워드 추가
        prompt_parts.extend([
            "professional album cover design",
            "artistic illustration",
            "high quality",
            "detailed artwork",
            "vibrant colors",
            "modern design",
            "trending on artstation"
        ])

        final_prompt = ", ".join(prompt_parts)

        # 네거티브 프롬프트 (SDXL Lightning 최적화)
        negative_prompt = "text, letters, words, watermark, signature, logo, blurry, low quality, ugly, deformed, distorted"

        return final_prompt, negative_prompt

    def generate_album_art(self, prompt, negative_prompt="", **generation_kwargs):
        """
        SDXL Lightning으로 앨범 아트 생성

        Args:
            prompt: 생성 프롬프트
            negative_prompt: 네거티브 프롬프트
            **generation_kwargs: 추가 생성 파라미터
        """
        if self.sd_pipeline is None:
            return None, "SDXL Lightning 모델이 로드되지 않았습니다"

        try:
            print(f"🎨 SDXL Lightning 이미지 생성 시작...")
            print(f"프롬프트: {prompt}")

            # 기본 생성 파라미터 (SDXL Lightning 최적화)
            default_params = {
                "num_inference_steps": self.lightning_steps,  # Lightning 스텝 수
                "guidance_scale": 0.0,  # Lightning은 0.0 권장
                "width": 1024,
                "height": 1024
            }

            # 사용자 파라미터로 덮어쓰기
            default_params.update(generation_kwargs)

            # 이미지 생성
            start_time = time.time()

            result = self.sd_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                **default_params
            )

            generation_time = time.time() - start_time

            print(f"✅ 이미지 생성 완료 ({generation_time:.1f}초)")

            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result.images[0], None

        except Exception as e:
            error_msg = f"이미지 생성 실패: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg

    def process_music_to_art(self, audio_file, **generation_kwargs):
        """
        전체 파이프라인: 음악 → 앨범 아트 (SDXL Lightning 기반)

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

        # 4. 이미지 생성 (SDXL Lightning)
        print("🔄 3단계: SDXL Lightning 앨범 아트 생성 중...")
        generated_image, error = self.generate_album_art(prompt, negative_prompt, **generation_kwargs)

        if error:
            return None, f"❌ {error}", analysis_text, prompt

        return generated_image, "✅ 앨범 아트 생성 완료! (SDXL Lightning)", analysis_text, prompt

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

        # Lightning 정보 추가
        text_parts.append(f"\n🚀 생성 모델: SDXL Lightning ({self.lightning_steps}스텝)")

        return "\n".join(text_parts)

    def change_lightning_steps(self, new_steps):
        """Lightning 스텝 수 변경"""
        if new_steps in [2, 4, 8]:
            print(f"🔄 Lightning 스텝 변경: {self.lightning_steps} → {new_steps}")
            self.lightning_steps = new_steps
            self._setup_sdxl_lightning()  # 모델 재로드
        else:
            print(f"❌ 지원하지 않는 스텝 수: {new_steps} (2, 4, 8만 지원)")


# 편의 함수들
def create_album_art_generator(music_classifier_path=None,
                               base_model="stabilityai/stable-diffusion-xl-base-1.0",
                               lightning_steps=4):
    """앨범 아트 생성기 인스턴스 생성 (SDXL Lightning)"""
    return AlbumArtGenerator(music_classifier_path, base_model, lightning_steps=lightning_steps)


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


# 사용 예시
if __name__ == "__main__":
    # 생성기 초기화
    generator = create_album_art_generator(lightning_steps=4)

    if generator.is_ready():
        print("✅ 앨범 아트 생성기 준비 완료!")
        print(f"📊 상태: {generator.get_status()}")
    else:
        print("❌ 생성기 초기화 실패")