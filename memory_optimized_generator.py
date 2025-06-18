# memory_optimized_generator.py
# T4 GPU용 초경량 앨범 아트 생성기

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gc
import psutil
from pathlib import Path


class LightweightAlbumArtGenerator:
    """T4 GPU 메모리 최적화 앨범 아트 생성기"""

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """
        초경량 생성기 초기화

        Args:
            model_id: 경량 모델 ID (SD 1.5 사용)
        """
        self.model_id = model_id
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🪶 초경량 생성기 초기화 중... (디바이스: {self.device})")
        self._setup_pipeline()

    def _setup_pipeline(self):
        """초경량 파이프라인 설정"""
        try:
            # 메모리 상태 체크
            self._check_memory()

            # SD 1.5 로드 (SDXL보다 훨씬 가벼움)
            print(f"📦 경량 모델 로딩: {self.model_id}")

            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )

            # 빠른 스케줄러로 교체
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            if self.device == "cuda":
                # 최대 메모리 절약 설정
                self.pipeline.enable_model_cpu_offload()  # 모델들을 CPU로 오프로드
                self.pipeline.enable_attention_slicing()  # 어텐션 슬라이싱
                self.pipeline.enable_vae_slicing()  # VAE 슬라이싱

                print("✅ 초경량 GPU 파이프라인 로드 완료")
            else:
                print("✅ 초경량 CPU 파이프라인 로드 완료")

            # 메모리 정리
            self._cleanup_memory()

        except Exception as e:
            print(f"❌ 초경량 파이프라인 로드 실패: {e}")
            self.pipeline = None

    def _check_memory(self):
        """메모리 상태 확인"""
        # RAM 확인
        memory = psutil.virtual_memory()
        print(f"📊 RAM 사용률: {memory.percent:.1f}% ({memory.used / 1e9:.1f}GB/{memory.total / 1e9:.1f}GB)")

        # GPU 메모리 확인
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"📊 GPU 메모리: {allocated:.1f}GB/{gpu_memory:.1f}GB 사용 중")

    def _cleanup_memory(self):
        """메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 메모리 정리 완료")

    def generate_image(self, prompt, negative_prompt="", num_steps=10,
                       guidance_scale=7.5, width=512, height=512, seed=-1):
        """
        경량 이미지 생성

        Args:
            num_steps: 10-20 스텝 (빠른 생성)
            width, height: 512x512 권장 (메모리 절약)
        """
        if self.pipeline is None:
            return None, "❌ 파이프라인이 로드되지 않았습니다"

        try:
            print(f"🪶 경량 이미지 생성 시작... ({width}x{height})")
            print(f"📝 프롬프트: {prompt}")

            # 메모리 정리
            self._cleanup_memory()

            # 시드 설정
            if seed != -1:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            # 이미지 생성 (경량 설정)
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_steps,  # 적은 스텝으로 빠르게
                guidance_scale=guidance_scale,
                width=width,
                height=height
            )

            # 생성 후 메모리 정리
            self._cleanup_memory()

            print("✅ 경량 이미지 생성 완료!")
            return result.images[0], None

        except Exception as e:
            error_msg = f"❌ 이미지 생성 실패: {str(e)}"
            print(error_msg)
            self._cleanup_memory()
            return None, error_msg

    def create_album_art_prompt(self, music_analysis, music_title="Unknown"):
        """음악 분석을 바탕으로 SD 1.5용 프롬프트 생성"""

        if "error" in music_analysis:
            return (f"album cover for {music_title}, artistic design, professional",
                    "text, words, blurry, low quality, ugly")

        # 기본 앨범 커버 스타일
        prompt_parts = [
            f"album cover for '{music_title}'",
            "professional music artwork",
            "artistic design",
            "high quality illustration"
        ]

        # 장르 정보 추가
        genres = music_analysis.get('genres', {}).get('top_genres', [])
        if genres:
            genre = genres[0]['genre']
            prompt_parts.append(f"{genre} music style")

        # 분위기 정보 추가
        moods = music_analysis.get('moods', {})
        prominent_moods = moods.get('prominent_moods', [])
        if prominent_moods:
            mood_text = ", ".join(prominent_moods[:2])
            prompt_parts.append(f"{mood_text} mood")
        elif moods.get('top_all'):
            top_mood = moods['top_all'][0][0]
            prompt_parts.append(f"{top_mood} atmosphere")

        # SD 1.5에 적합한 스타일 키워드
        prompt_parts.extend([
            "detailed artwork",
            "vibrant colors",
            "digital art",
            "trending on artstation"
        ])

        final_prompt = ", ".join(prompt_parts)
        negative_prompt = "text, words, letters, watermark, signature, blurry, low quality, ugly, deformed"

        return final_prompt, negative_prompt


# Gradio 앱용 경량 래퍼
class LightweightGradioApp:
    """경량 Gradio 앱"""

    def __init__(self, music_classifier=None):
        self.generator = LightweightAlbumArtGenerator()
        self.music_classifier = music_classifier

    def process_music_to_art(self, audio_file, num_steps=10, width=512, height=512):
        """경량 음악→아트 파이프라인"""

        if audio_file is None:
            return None, "❌ 음악 파일을 업로드해주세요", "", ""

        music_title = Path(audio_file).stem

        # 1. 음악 분석 (분류기가 있으면)
        if self.music_classifier:
            print("🔄 음악 분석 중...")
            with open(audio_file, 'rb') as f:
                audio_data = f.read()

            # 임시 파일로 분석
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name

            try:
                music_result = self.music_classifier.classify_music(tmp_path)
                analysis_text = self._format_analysis(music_result)
                os.unlink(tmp_path)
            except:
                music_result = {"error": "분석 실패"}
                analysis_text = "음악 분석 실패"
        else:
            music_result = {"error": "분류기 없음"}
            analysis_text = "음악 분류기가 로드되지 않았습니다"

        # 2. 프롬프트 생성
        prompt, negative_prompt = self.generator.create_album_art_prompt(music_result, music_title)

        # 3. 이미지 생성
        print("🔄 경량 이미지 생성 중...")
        image, error = self.generator.generate_image(
            prompt, negative_prompt, num_steps, width=width, height=height
        )

        if error:
            return None, f"❌ {error}", analysis_text, prompt

        return image, "✅ 경량 앨범 아트 생성 완료!", analysis_text, prompt

    def _format_analysis(self, result):
        """분석 결과 포맷팅"""
        if "error" in result:
            return f"오류: {result['error']}"

        text_parts = []
        duration = result.get('audio_duration', 0)
        text_parts.append(f"⏱️ 길이: {duration:.1f}초")

        genres = result.get('genres', {}).get('top_genres', [])
        if genres:
            text_parts.append(f"\n🎼 상위 장르: {genres[0]['genre']}")

        moods = result.get('moods', {}).get('top_all', [])
        if moods:
            text_parts.append(f"🎭 주요 분위기: {moods[0][0]}")

        return "\n".join(text_parts)


# 사용 예시
def create_lightweight_app(music_classifier=None):
    """경량 앱 생성"""
    import gradio as gr

    app = LightweightGradioApp(music_classifier)

    with gr.Blocks(title="🪶 경량 앨범 아트 생성기") as demo:
        gr.Markdown("""
        # 🪶 경량 앨범 아트 생성기
        ## T4 GPU 메모리 최적화 버전

        - **모델**: Stable Diffusion 1.5 (경량)
        - **해상도**: 512x512 (메모리 절약)
        - **속도**: 10-20 스텝 (빠른 생성)
        """)

        with gr.Row():
            with gr.Column():
                audio_input = gr.File(label="🎵 음악 파일", file_types=[".mp3", ".wav"])

                with gr.Row():
                    width_input = gr.Slider(label="너비", minimum=256, maximum=768, step=64, value=512)
                    height_input = gr.Slider(label="높이", minimum=256, maximum=768, step=64, value=512)

                steps_input = gr.Slider(label="생성 스텝", minimum=5, maximum=25, step=5, value=10)
                generate_btn = gr.Button("🪶 경량 생성", variant="primary")

            with gr.Column():
                status_output = gr.Textbox(label="상태", interactive=False)
                image_output = gr.Image(label="생성 이미지", height=400)

        with gr.Accordion("분석 결과", open=False):
            analysis_output = gr.Textbox(label="음악 분석", lines=5)
            prompt_output = gr.Textbox(label="생성 프롬프트", lines=3)

        generate_btn.click(
            fn=app.process_music_to_art,
            inputs=[audio_input, steps_input, width_input, height_input],
            outputs=[image_output, status_output, analysis_output, prompt_output]
        )

    return demo


# 직접 실행
if __name__ == "__main__":
    demo = create_lightweight_app()
    demo.launch(share=True)