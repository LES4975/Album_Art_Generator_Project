# gradio_app.py
# Gradio 인터페이스 전용 파일

import gradio as gr
import time
from pathlib import Path

# 핵심 모듈 임포트
from album_art_core import AlbumArtGenerator, extract_music_title


class GradioAlbumArtApp:
    """Gradio 앨범 아트 앱 클래스"""

    def __init__(self, music_classifier_path=None, sd_model_id="runwayml/stable-diffusion-v1-5",
                 classifier_instance=None):
        """
        초기화

        Args:
            music_classifier_path: 음악 분류기 모듈 경로 (선택사항)
            sd_model_id: Stable Diffusion 모델 ID
            classifier_instance: 이미 초기화된 음악 분류기 인스턴스 (선택사항)
        """
        print("🚀 앨범 아트 생성기 초기화 중...")

        # 이미 초기화된 분류기가 있으면 사용, 없으면 새로 생성
        if classifier_instance:
            self.art_generator = AlbumArtGenerator(music_classifier_path, sd_model_id)
            self.art_generator.music_classifier = classifier_instance
            print("✅ 기존 음악 분류기 인스턴스 사용")
        else:
            self.art_generator = AlbumArtGenerator(music_classifier_path, sd_model_id)

        # 상태 확인
        status = self.art_generator.get_status()
        print(f"📊 모델 상태: {status}")

    def on_file_upload(self, audio_file):
        """파일 업로드 시 제목 자동 감지"""
        if audio_file is None:
            return ""

        music_title = extract_music_title(audio_file)
        return music_title

    def on_generate_click(self, audio_file):
        """생성 버튼 클릭 시 실행"""

        if audio_file is None:
            return None, "❌ 음악 파일을 업로드해주세요", "", "", "", gr.update(visible=False)

        # 파일명에서 제목 추출
        music_title = extract_music_title(audio_file)

        # 진행 상태 업데이트
        yield None, "🔄 음악 분석 중...", "", "", music_title, gr.update(visible=False)

        # 앨범 아트 생성
        image, status, analysis, prompt = self.art_generator.process_music_to_art(audio_file)

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
        """Gradio 인터페이스 생성"""

        with gr.Blocks(title="🎵 Music-Based Album Art Generator", theme=gr.themes.Soft()) as demo:
            # 헤더
            gr.Markdown("""
            # 🎵 Music-Based Album Art Generator

            음악을 업로드하면 자동으로 분석하여 어울리는 앨범 아트를 생성합니다!

            **사용 방법:**
            1. 음악 파일을 업로드하세요 (MP3, WAV 등)
            2. 파일명이 곡 제목으로 자동 감지됩니다
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
                        label="🎨 생성된 앨범 아트",
                        type="pil",
                        height=400
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
                    label="📝 생성된 프롬프트",
                    lines=3,
                    interactive=False
                )

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
                inputs=[audio_input],
                outputs=[image_output, status_output, analysis_output, prompt_output, title_display, download_btn]
            )

            # 예시 섹션
            with gr.Row():
                gr.Markdown("""
                ### 💡 팁
                - **지원 형식**: MP3, WAV, M4A, FLAC
                - **권장 길이**: 30초 이상 (더 정확한 분석)
                - **파일명 자동 감지**: 업로드된 파일명이 곡 제목으로 사용됩니다
                - **생성 시간**: GPU 사용시 약 10-30초, CPU 사용시 1-3분

                ### 🎯 프로젝트 정보
                - **음악 분석**: Essentia + 사전학습된 모델
                - **이미지 생성**: Stable Diffusion v1.5
                - **개발 환경**: Google Colab
                """)

        return demo

    def launch(self, **launch_kwargs):
        """앱 실행"""
        demo = self.create_interface()

        print("🚀 앱 시작!")
        print("=" * 50)

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
def create_and_launch_app(music_classifier_path=None, sd_model_id="runwayml/stable-diffusion-v1-5",
                          classifier_instance=None, **launch_kwargs):
    """앱 생성 및 실행"""
    app = GradioAlbumArtApp(music_classifier_path, sd_model_id, classifier_instance)
    return app.launch(**launch_kwargs)


# 직접 실행시
if __name__ == "__main__":
    # Google Colab에서 실행
    create_and_launch_app()