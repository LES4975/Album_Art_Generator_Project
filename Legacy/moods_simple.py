import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
import requests
import warnings
warnings.filterwarnings('ignore')

class CorrectedMusicMoodClassifier:
    def __init__(self, model_dir="models"):
        """
        분석 결과를 바탕으로 수정된 음악 분위기 분류기
        실제 텐서 이름과 모양을 사용
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # 모델 파일 경로
        self.embeddings_model_path = self.model_dir / "discogs-effnet-bs64-1.pb"
        self.mood_model_path = self.model_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"

        # 분석 결과에서 확인된 정확한 텐서 이름과 모양
        self.embeddings_input_name = "serving_default_melspectrogram:0"
        self.embeddings_output_name = "PartitionedCall:1"
        self.embeddings_input_shape = [64, 128, 96]  # [batch, mel_bins, time_frames]
        self.embeddings_output_shape = [64, 1280]    # [batch, embedding_dim]

        self.mood_input_name = "model/Placeholder:0"
        self.mood_output_name = "model/Sigmoid:0"
        self.mood_input_shape = [None, 1280]         # [batch, embedding_dim]
        self.mood_output_shape = [None, 56]          # [batch, num_tags]

        # 56개 태그 정의 (Essentia 참조 코드와 동일)
        self.all_tags = [
            "action", "adventure", "advertising",
            "background", "ballad",
            "calm", "children", "christmas", "commercial", "cool", "corporate",
            "dark", "deep", "documentary", "drama", "dramatic", "dream",
            "emotional", "energetic", "epic",
            "fast", "film", "fun", "funny",
            "game", "groovy",
            "happy", "heavy", "holiday", "hopeful",
            "inspiring",
            "love",
            "meditative", "melancholic", "melodic", "motivational", "movie",
            "nature",
            "party", "positive", "powerful",
            "relaxing", "retro", "romantic",
            "sad", "sexy", "slow", "soft", "soundscape", "space", "sport", "summer",
            "trailer", "travel",
            "upbeat", "uplifting"
        ]

        # 카테고리별 분류
        self.mood_tags = [
            "calm", "cool", "dark", "deep", "dramatic",
            "emotional", "energetic", "epic", "fast", "fun", "funny",
            "groovy", "happy", "heavy", "hopeful", "inspiring",
            "meditative", "melancholic", "motivational",
            "positive", "powerful", "relaxing", "romantic",
            "sad", "sexy", "slow", "soft", "upbeat", "uplifting"
        ]

        self.theme_tags = [
            "action", "adventure", "ballad", "children", "christmas",
            "dream", "film", "game", "holiday", "love", "movie",
            "nature", "party", "retro", "space", "sport", "summer", "travel"
        ]

        self.function_tags = [
            "advertising", "background", "commercial", "corporate",
            "documentary", "drama", "soundscape", "trailer"
        ]

        # TensorFlow 세션들
        self.embeddings_session = None
        self.embeddings_graph = None
        self.mood_session = None
        self.mood_graph = None

        # 모델 다운로드 및 로드
        self._download_models()
        self._load_tensorflow_models()

    def _download_models(self):
        """모델 파일들을 다운로드"""
        models_to_download = [
            {
                "url": "https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb",
                "path": self.embeddings_model_path,
                "name": "Discogs EfficientNet"
            },
            {
                "url": "https://essentia.upf.edu/models/mood-style-classification/jamendo-mood-theme-classes/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
                "path": self.mood_model_path,
                "name": "MTG Jamendo Mood Theme"
            }
        ]

        for model in models_to_download:
            if not model["path"].exists():
                print(f"📥 {model['name']} 다운로드 중...")
                try:
                    response = requests.get(model["url"], stream=True)
                    response.raise_for_status()

                    with open(model["path"], "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    print(f"✅ {model['name']} 다운로드 완료!")
                except Exception as e:
                    print(f"❌ {model['name']} 다운로드 실패: {e}")
            else:
                print(f"✅ {model['name']} 이미 존재함")

    def _load_tensorflow_models(self):
        """TensorFlow 모델들을 로드"""
        try:
            # 임베딩 모델 로드
            if self.embeddings_model_path.exists():
                print("🔄 임베딩 모델 로드 중...")
                self.embeddings_graph, self.embeddings_session = self._load_pb_model(self.embeddings_model_path)
                if self.embeddings_session is not None:
                    print("✅ 임베딩 모델 로드 완료!")

            # 분위기 분류 모델 로드
            if self.mood_model_path.exists():
                print("🔄 분위기 분류 모델 로드 중...")
                self.mood_graph, self.mood_session = self._load_pb_model(self.mood_model_path)
                if self.mood_session is not None:
                    print("✅ 분위기 분류 모델 로드 완료!")

        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")

    def _load_pb_model(self, model_path):
        """단일 .pb 파일을 로드"""
        try:
            # GraphDef 로드
            with tf.io.gfile.GFile(str(model_path), "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

            # 새 그래프 생성
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(graph_def, name="")

            # 세션 생성
            session = tf.compat.v1.Session(graph=graph)

            return graph, session

        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return None, None

    def preprocess_audio_for_discogs(self, audio, sr=16000):
        """
        Discogs 모델에 맞는 멜 스펙트로그램 생성
        분석 결과: [64, 128, 96] 형태 필요
        """
        try:
            # 1. 리샘플링 (16kHz)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # 2. 배치 크기 64에 맞게 오디오 분할
            # 전체 길이를 64개 세그먼트로 나누기
            target_duration_per_segment = len(audio) / 64  # 각 세그먼트의 샘플 수
            segments = []

            for i in range(64):
                start_idx = int(i * target_duration_per_segment)
                end_idx = int((i + 1) * target_duration_per_segment)
                segment = audio[start_idx:end_idx]

                if len(segment) == 0:
                    segments.append(np.zeros(int(target_duration_per_segment)))
                else:
                    segments.append(segment)

            # 3. 각 세그먼트에 대해 멜 스펙트로그램 계산
            mel_spectrograms = []

            for segment in segments:
                if len(segment) > 0:
                    # 멜 스펙트로그램 계산
                    mel_spec = librosa.feature.melspectrogram(
                        y=segment,
                        sr=sr,
                        n_mels=128,        # 분석 결과에서 확인된 값
                        n_fft=2048,
                        hop_length=512,
                        fmin=0,
                        fmax=sr/2
                    )

                    # 로그 스케일 변환
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                    # 시간 축을 96으로 맞추기
                    if mel_spec_db.shape[1] > 96:
                        mel_spec_db = mel_spec_db[:, :96]
                    elif mel_spec_db.shape[1] < 96:
                        pad_width = 96 - mel_spec_db.shape[1]
                        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')

                    mel_spectrograms.append(mel_spec_db)
                else:
                    # 빈 세그먼트의 경우 영행렬
                    mel_spectrograms.append(np.zeros((128, 96)))

            # 4. 배치로 결합: [64, 128, 96]
            mel_batch = np.array(mel_spectrograms, dtype=np.float32)

            # 5. 정규화
            mel_batch = (mel_batch - mel_batch.mean()) / (mel_batch.std() + 1e-8)

            print(f"멜 스펙트로그램 형태: {mel_batch.shape}")
            return mel_batch

        except Exception as e:
            print(f"오디오 전처리 실패: {e}")
            return None

    def extract_embeddings(self, mel_spectrogram):
        """
        Discogs 모델로 임베딩 추출
        """
        if self.embeddings_session is None or mel_spectrogram is None:
            return None

        try:
            # 입력/출력 텐서 가져오기
            input_tensor = self.embeddings_graph.get_tensor_by_name(self.embeddings_input_name)
            output_tensor = self.embeddings_graph.get_tensor_by_name(self.embeddings_output_name)

            # 추론 실행
            embeddings = self.embeddings_session.run(
                output_tensor,
                feed_dict={input_tensor: mel_spectrogram}
            )

            print(f"임베딩 형태: {embeddings.shape}")
            return embeddings

        except Exception as e:
            print(f"임베딩 추출 실패: {e}")
            print(f"입력 텐서: {self.embeddings_input_name}")
            print(f"출력 텐서: {self.embeddings_output_name}")
            print(f"입력 형태: {mel_spectrogram.shape if mel_spectrogram is not None else 'None'}")
            return None

    def classify_mood(self, embeddings):
        """
        임베딩으로부터 분위기 분류
        """
        if self.mood_session is None or embeddings is None:
            return None

        try:
            # 임베딩 평균내기 (배치 차원 제거)
            if len(embeddings.shape) > 1:
                embeddings_mean = np.mean(embeddings, axis=0, keepdims=True)  # [1, 1280]
            else:
                embeddings_mean = embeddings.reshape(1, -1)

            # 정규화 추가
            embeddings_mean = (embeddings_mean - embeddings_mean.mean()) / embeddings_mean.std()

            # 입력/출력 텐서 가져오기
            input_tensor = self.mood_graph.get_tensor_by_name(self.mood_input_name)
            output_tensor = self.mood_graph.get_tensor_by_name(self.mood_output_name)

            # 추론 실행
            predictions = self.mood_session.run(
                output_tensor,
                feed_dict={input_tensor: embeddings_mean}
            )

            # 배치 차원 제거
            if len(predictions.shape) > 1:
                predictions = predictions[0]

            print(f"예측 결과 형태: {predictions.shape}")

            # 태그별 딕셔너리 생성
            activations_dict = {}
            for i, tag in enumerate(self.all_tags):
                if i < len(predictions):
                    activations_dict[tag] = float(predictions[i])
                else:
                    activations_dict[tag] = 0.0

            return activations_dict

        except Exception as e:
            print(f"분위기 분류 실패: {e}")
            print(f"입력 텐서: {self.mood_input_name}")
            print(f"출력 텐서: {self.mood_output_name}")
            print(f"임베딩 형태: {embeddings.shape if embeddings is not None else 'None'}")
            return None

    def predict_mood(self, file_path, method="smart_segment"):
        """
        음악 파일의 분위기 예측 (실제 모델 사용)
        """
        print(f"🎵 분석 시작: {file_path}")
        print(f"전처리 방법: {method}")
        print("=" * 50)

        # 1. 오디오 로드
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            print(f"✅ 오디오 로드 완료: {len(audio)/sr:.1f}초")
        except Exception as e:
            return {"error": f"오디오 로드 실패: {e}"}

        # 2. 전처리 (멜 스펙트로그램 생성)
        print("🔄 멜 스펙트로그램 생성 중...")
        mel_spectrogram = self.preprocess_audio_for_discogs(audio, sr)
        if mel_spectrogram is None:
            return {"error": "오디오 전처리 실패"}

        # 3. 임베딩 추출
        print("🔄 임베딩 추출 중...")
        embeddings = self.extract_embeddings(mel_spectrogram)
        if embeddings is None:
            return {"error": "임베딩 추출 실패"}

        # 4. 분위기 분류
        print("🔄 분위기 분류 중...")
        activations = self.classify_mood(embeddings)
        if activations is None:
            return {"error": "분위기 분류 실패"}

        # 5. 결과 처리
        try:
            print("🔄 결과 처리 중...")

            # 아웃라이어 임계값 계산 (Essentia 방식과 동일)
            values = list(activations.values())
            q1 = np.quantile(values, 0.25)
            q3 = np.quantile(values, 0.75)
            outlier_threshold = q3 + (1.5 * (q3 - q1))

            print(f"임계값: {outlier_threshold:.3f}")

            # 임계값 이상의 태그 선택
            prominent_tags = [
                tag for tag, score in activations.items()
                if score >= outlier_threshold and tag != 'melodic'
            ]

            # 카테고리별 분류
            moods = [tag for tag in prominent_tags if tag in self.mood_tags]
            themes = [tag for tag in prominent_tags if tag in self.theme_tags]
            functions = [tag for tag in prominent_tags if tag in self.function_tags]

            # 상위 태그들
            sorted_tags = sorted(activations.items(), key=lambda x: x[1], reverse=True)
            top_tags = sorted_tags[:10]

            return {
                "file_path": file_path,
                "preprocessing_method": method,
                "primary_moods": moods,
                "themes": themes,
                "functions": functions,
                "top_tags": top_tags,
                "all_activations": activations,
                "model_used": "Essentia Discogs-EfficientNet (Direct TensorFlow)",
                "outlier_threshold": outlier_threshold
            }

        except Exception as e:
            return {"error": f"결과 처리 실패: {e}"}

    def close_sessions(self):
        """TensorFlow 세션들을 정리"""
        if self.embeddings_session:
            self.embeddings_session.close()
        if self.mood_session:
            self.mood_session.close()
        print("✅ TensorFlow 세션들이 정리되었습니다.")

def main():
    """
    메인 실행 함수
    """
    classifier = CorrectedMusicMoodClassifier()

    # 음악 파일 경로
    music_file = "../musics/Blood_-_All_My_Friends_Hate_Me.mp3"  # 실제 파일 경로로 변경

    print(f"🎵 실제 Discogs 모델을 사용한 분위기 분석")
    print(f"📁 파일: {music_file}")
    print("=" * 70)

    try:
        # 분위기 예측
        result = classifier.predict_mood(music_file)

        # 결과 출력
        if "error" in result:
            print(f"❌ 오류: {result['error']}")
        else:
            print(f"\n✅ 분석 완료!")
            print(f"🤖 모델: {result['model_used']}")
            print(f"📊 임계값: {result['outlier_threshold']:.3f}")
            print()

            print("🎭 주요 분위기:")
            if result['primary_moods']:
                for mood in result['primary_moods']:
                    print(f"  • {mood}")
            else:
                print("  (임계값을 넘는 분위기 없음)")

            print(f"\n🎨 테마:")
            if result['themes']:
                for theme in result['themes']:
                    print(f"  • {theme}")
            else:
                print("  (임계값을 넘는 테마 없음)")

            print(f"\n⚙️  기능:")
            if result['functions']:
                for function in result['functions']:
                    print(f"  • {function}")
            else:
                print("  (임계값을 넘는 기능 없음)")

            print(f"\n📊 상위 태그 (신뢰도순):")
            for tag, score in result['top_tags']:
                print(f"  • {tag}: {score:.3f}")

    finally:
        # 세션 정리
        classifier.close_sessions()

if __name__ == "__main__":
    print("🎵 === 수정된 TensorFlow Discogs 모델 사용 ===")
    print("분석 결과를 바탕으로 정확한 텐서 이름 사용")
    print()

    try:
        main()
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다.")