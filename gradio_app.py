# gradio_app.py
# SDXL Lightning 기반 Gradio 인터페이스 (업데이트 버전)

import gradio as gr
import time
from pathlib import Path
import torch

# 핵심 모듈 임포트
from album_art_core import AlbumArtGenerator, extract_music_title


class GradioAlbumArtApp:
    """SDXL Lightning 기반 Gradio 앨범 아트 앱 클래스"""

    def __init__(self, music_classifier_path=None,
                 base_model="stabilityai/stable-diffusion-xl-base-1.0",
                 lightning_steps=4,
                 classifier_instance=None):
        """
        초기화

        Args:
            music_classifier_path: 음악 분류기 모듈 경로 (선택사항)
            base_model: SDXL 기본 모델 ID
            lightning_steps: Lightning 스텝 수 (2, 4, 8)
            classifier_instance: 이미 초기화된 음악 분류기 인스턴스 (선택사항)
        """
        print("🚀 SDXL Lightning 앨범 아트 생성기 초기화 중...")

        # 이미 초기화된 분류기가 있으면 사용, 없으면 새로 생성
        if classifier_instance:
            self.art_generator = AlbumArtGenerator(
                music_classifier_path, base_model, lightning_steps=lightning_steps
            )
            self.art_generator.music_classifier = classifier_instance
            print("✅ 기존 음악 분류기 인스턴스 사용")
        else:
            self.art_generator = AlbumArtGenerator(
                music_classifier_path, base_model, lightning_steps=lightning_steps
            )

        # 상태 확인
        status = self.art_generator.get_status()
        print(f"📊 모델 상태: {status}")

    def on_file_upload(self, audio_file):
        """파일 업로드 시 제목 자동 감지"""
        if audio_file is None:
            return ""

        music_title = extract_music_title(audio_file)
        return music_title

    def on_steps_change(self, new_steps):
        """Lightning 스텝 수 변경"""
        try:
            if new_steps in [2, 4, 8]:
                self.art_generator.change_lightning_steps(new_steps)
                return f"✅ Lightning 스텝이 {new_steps}로 변경되었습니다"
            else:
                return "❌ 2, 4, 8 스텝만 지원됩니다"
        except Exception as e:
            return f"❌ 스텝 변경 실패: {str(e)}"

    def on_generate_click(self, audio_file, lightning_steps, width, height):
        """생성 버튼 클릭 시 실행"""

        if audio_file is None:
            return None, "❌ 음악 파일을 업로드해주세요", "", "", "", gr.update(visible=False)

        # Lightning 스텝 수 변경 (필요시)
        if lightning_steps != self.art_generator.lightning_steps:
            step_result = self.on_steps_change(lightning_steps)
            print(step_result)

        # 파일명에서 제목 추출
        music_title = extract_music_title(audio_file)

        # 진행 상태 업데이트
        yield None, "🔄 음악 분석 중...", "", "", music_title, gr.update(visible=False)

        # 앨범 아트 생성 (해상도 설정 포함)
        generation_kwargs = {
            "width": width,
            "height": height
        }

        image, status, analysis, prompt = self.art_generator.process_music_to_art(
            audio_file, **generation_kwargs
        )

        # 다운로드 버튼 준비
        download_visible = image is not None
        download_update = gr.update(visible=download_visible)

        if image:
            # 이미지를 임시 파일로 저장 (다운로드용)
            temp_path = f"/tmp/album_art_{music_title}_{int(time.time())}.png"
            image.save(temp_path)
            download_update = gr.update(visible=True, value=temp_path)

        return image, status, analysis, prompt, music_title, download_update

    def create_interface(self):
        """SDXL Lightning Gradio 인터페이스 생성"""

        with gr.Blocks(title="🎵 SDXL Lightning Album Art Generator", theme=gr.themes.Soft()) as demo:
            # 헤더
            gr.Markdown("""
            # 🎵 Music-Based Album Art Generator
            ## ⚡ Powered by SDXL Lightning

            음악을 업로드하면 자동으로 분석하여 어울리는 고품질 앨범 아트를 생성합니다!

            **🚀 SDXL Lightning 특징:**
            - **고품질**: 1024x1024 해상도 지원
            - **고속도**: 2-8 스텝으로 빠른 생성
            - **균형**: 품질과 속도의 최적 균형

            **사용 방법:**
            1. 음악 파일을 업로드하세요 (MP3, WAV 등)
            2. Lightning 설정을 조정하세요 (빠름 ↔ 고품질)
            3. "앨범 아트 생성" 버튼을 클릭하세요
            4. 결과를 확인하고 다운로드하세요!
            """)

            with gr.Row():
                # 입력 섹션
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 입력")

                    audio_input = gr.File(
                        label="🎵 음악 파일 업로드",
                        file_types=[".mp3", ".wav", ".m4a", ".flac"],
                        type="filepath"
                    )

                    title_display = gr.Textbox(
                        label="🎼 감지된 곡 제목",
                        placeholder="파일 업로드 시 자동 감지됩니다",
                        interactive=False,
                        value=""
                    )

                    # Lightning 설정
                    gr.Markdown("### ⚡ Lightning 설정")

                    lightning_steps = gr.Radio(
                        label="🔄 생성 스텝 (품질 vs 속도)",
                        choices=[2, 4, 8],
                        value=4,
                        info="2: 초고속, 4: 균형, 8: 고품질"
                    )

                    # 이미지 해상도 설정
                    gr.Markdown("### 📏 이미지 설정")

                    with gr.Row():
                        width_input = gr.Slider(
                            label="너비",
                            minimum=512,
                            maximum=1536,
                            step=64,
                            value=1024
                        )
                        height_input = gr.Slider(
                            label="높이",
                            minimum=512,
                            maximum=1536,
                            step=64,
                            value=1024
                        )

                    generate_btn = gr.Button(
                        "🎨 앨범 아트 생성",
                        variant="primary",
                        size="lg"
                    )

                # 출력 섹션
                with gr.Column(scale=2):
                    gr.Markdown("### 🖼️ 결과")

                    status_output = gr.Textbox(
                        label="📋 상태",
                        value="음악 파일을 업로드하고 생성 버튼을 눌러주세요",
                        interactive=False
                    )

                    image_output = gr.Image(
                        label="🎨 생성된 앨범 아트 (SDXL Lightning)",
                        type="pil",
                        height=500
                    )

                    download_btn = gr.DownloadButton(
                        label="💾 앨범 아트 다운로드",
                        visible=False
                    )

            # 상세 정보 섹션 (접을 수 있음)
            with gr.Accordion("🔍 상세 분석 결과", open=False):
                analysis_output = gr.Textbox(
                    label="📊 음악 분석 결과",
                    lines=10,
                    interactive=False
                )

                prompt_output = gr.Textbox(
                    label="📝 생성된 프롬프트 (SDXL Lightning 최적화)",
                    lines=3,
                    interactive=False
                )

            # Lightning 설정 정보
            with gr.Accordion("⚡ SDXL Lightning 정보", open=False):
                gr.Markdown("""
                ### 🔧 Lightning 스텝 설명

                | 스텝 | 속도 | 품질 | 권장 용도 |
                |------|------|------|-----------|
                | **2스텝** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 빠른 프리뷰, 실시간 생성 |
                | **4스텝** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | **균형 잡힌 기본 설정** |
                | **8스텝** | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 최고 품질, 최종 결과물 |

                ### 📏 해상도 권장사항
                - **1024x1024**: 정사각형 앨범 커버 (기본)
                - **1024x768**: 가로형 레이아웃
                - **768x1024**: 세로형 레이아웃
                - **1536x1536**: 초고해상도 (메모리 주의)

                ### 💡 최적화 팁
                - T4 GPU: 4스텝 + 1024x1024 권장
                - V100/A100: 8스텝 + 1536x1536 가능
                - 메모리 부족시: 해상도를 낮추세요
                """)

            # 이벤트 연결
            # 파일 업로드 시 제목 자동 감지
            audio_input.change(
                fn=self.on_file_upload,
                inputs=[audio_input],
                outputs=[title_display]
            )

            # 버튼 이벤트 연결
            generate_btn.click(
                fn=self.on_generate_click,
                inputs=[audio_input, lightning_steps, width_input, height_input],
                outputs=[image_output, status_output, analysis_output, prompt_output, title_display, download_btn]
            )

            # 예시 섹션
            with gr.Row():
                gr.Markdown("""
                ### 💡 사용 팁
                - **지원 형식**: MP3, WAV, M4A, FLAC
                - **권장 길이**: 30초 이상 (더 정확한 분석)
                - **파일명 자동 감지**: 업로드된 파일명이 곡 제목으로 사용됩니다
                - **Lightning 장점**: Turbo 대비 2배 좋은 품질, 일반 SDXL 대비 10배 빠른 속도

                ### 🎯 프로젝트 정보
                - **음악 분석**: Essentia + 사전학습된 모델 (장르, 분위기 분석)
                - **이미지 생성**: SDXL Lightning (ByteDance)
                - **최적화**: T4 GPU 메모리 효율성 향상
                - **개발 환경**: Google Colab

                ### 🚀 성능 비교
                | 모델 | 해상도 | 스텝 | T4에서 속도 | 품질 |
                |------|--------|------|-------------|------|
                | SDXL Lightning | 1024² | 4 | ~5초 | ⭐⭐⭐⭐⭐ |
                | SDXL Turbo | 512² | 1 | ~2초 | ⭐⭐⭐ |
                | SDXL 1.0 | 1024² | 50 | ~30초+ | ⭐⭐⭐⭐⭐ |
                """)

        return demo

    def launch(self, **launch_kwargs):
        """앱 실행"""
        demo = self.create_interface()

        print("🚀 SDXL Lightning 앱 시작!")
        print("=" * 50)

        # GPU 정보 출력
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
            print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠️ CPU 모드 (속도 매우 느림)")

        # 기본 launch 설정
        default_kwargs = {
            "share": True,  # 공유 링크 생성
            "debug": True,
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "show_error": True
        }

        # 사용자 설정으로 덮어쓰기
        default_kwargs.update(launch_kwargs)

        return demo.launch(**default_kwargs)


# 간편 실행 함수
def create_and_launch_app(music_classifier_path=None,
                          base_model="stabilityai/stable-diffusion-xl-base-1.0",
                          lightning_steps=4,
                          classifier_instance=None,
                          **launch_kwargs):
    """SDXL Lightning 앱 생성 및 실행"""
    app = GradioAlbumArtApp(
        music_classifier_path, base_model, lightning_steps, classifier_instance
    )
    return app.launch(**launch_kwargs)


# 직접 실행시
if __name__ == "__main__":
    # Google Colab에서 실행
    print("🎵 SDXL Lightning Album Art Generator")
    print("=" * 50)
    create_and_launch_app()