import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
import requests
import warnings
import json

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
        self.genre_model_path = self.model_dir / "genre_discogs400-discogs-effnet-1.pb"

        # JSON 메타데이터 파일 경로
        self.embeddings_json_path = self.model_dir / "discogs-effnet-bs64-1.json"
        self.mood_json_path = self.model_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.json"
        self.genre_json_path = self.model_dir / "genre_discogs400-discogs-effnet-1.json"

        # 분석 결과에서 확인된 정확한 텐서 이름과 모양
        self.embeddings_input_name = "serving_default_melspectrogram:0"
        self.embeddings_output_name = "PartitionedCall:1"
        self.embeddings_input_shape = [64, 128, 96]  # [batch, mel_bins, time_frames]
        self.embeddings_output_shape = [64, 1280]  # [batch, embedding_dim]

        self.mood_input_name = "model/Placeholder:0"
        self.mood_output_name = "model/Sigmoid:0"
        self.mood_input_shape = [None, 1280]  # [batch, embedding_dim]
        self.mood_output_shape = [None, 56]  # [batch, num_tags]

        # 장르 모델 텐서 정보
        self.genre_input_name = "serving_default_model_Placeholder:0"  # :0 복구
        self.genre_output_name = "PartitionedCall:0"
        self.genre_input_shape = [None, 1280]  # [batch, embedding_dim]
        self.genre_output_shape = [None, 400]  # [batch, num_genres]

        # 클래스 리스트들 (JSON에서 로드)
        self.genre_classes = []
        self.mood_classes = []
        self.embeddings_classes = []

        # 카테고리별 분류 (동적으로 생성됨)
        self.mood_tags = []
        self.theme_tags = []
        self.function_tags = []

        # TensorFlow 세션들
        self.embeddings_session = None
        self.embeddings_graph = None
        self.mood_session = None
        self.mood_graph = None
        self.genre_session = None
        self.genre_graph = None

        # JSON 메타데이터 다운로드 및 로드
        self._download_metadata()
        self._load_metadata()

        # 모델 다운로드 및 로드
        self._download_models()
        self._load_tensorflow_models()

    def _download_metadata(self):
        """JSON 메타데이터 파일들을 다운로드"""
        metadata_to_download = [
            {
                "url": "https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.json",
                "path": self.embeddings_json_path,
                "name": "Discogs EfficientNet Metadata"
            },
            {
                "url": "https://essentia.upf.edu/models/mood-style-classification/jamendo-mood-theme-classes/mtg_jamendo_moodtheme-discogs-effnet-1.json",
                "path": self.mood_json_path,
                "name": "MTG Jamendo Mood Theme Metadata"
            },
            {
                "url": "https://essentia.upf.edu/models/music-style-classification/jamendo-genre-classes/genre_discogs400-discogs-effnet-1.json",
                "path": self.genre_json_path,
                "name": "Discogs 400 Genre Metadata"
            }
        ]

        for metadata in metadata_to_download:
            if not metadata["path"].exists():
                print(f"📥 {metadata['name']} 다운로드 중...")
                try:
                    response = requests.get(metadata["url"], stream=True)
                    response.raise_for_status()

                    with open(metadata["path"], "w", encoding="utf-8") as f:
                        f.write(response.text)

                    print(f"✅ {metadata['name']} 다운로드 완료!")
                except Exception as e:
                    print(f"❌ {metadata['name']} 다운로드 실패: {e}")
            else:
                print(f"✅ {metadata['name']} 이미 존재함")

    def _load_metadata(self):
        """JSON 메타데이터에서 클래스 정보 로드"""
        try:
            # 장르 클래스 로드
            if self.genre_json_path.exists():
                with open(self.genre_json_path, 'r', encoding='utf-8') as f:
                    genre_metadata = json.load(f)
                    self.genre_classes = genre_metadata.get('classes', [])
                    print(f"✅ 장르 클래스 {len(self.genre_classes)}개 로드 완료!")

            # 분위기 클래스 로드
            if self.mood_json_path.exists():
                with open(self.mood_json_path, 'r', encoding='utf-8') as f:
                    mood_metadata = json.load(f)
                    self.mood_classes = mood_metadata.get('classes', [])
                    print(f"✅ 분위기 클래스 {len(self.mood_classes)}개 로드 완료!")

                    # 카테고리별 분류 동적 생성
                    self._categorize_mood_tags()

            # 임베딩 모델 메타데이터 로드
            if self.embeddings_json_path.exists():
                with open(self.embeddings_json_path, 'r', encoding='utf-8') as f:
                    embeddings_metadata = json.load(f)
                    self.embeddings_classes = embeddings_metadata.get('classes', [])
                    print(f"✅ 임베딩 모델 메타데이터 로드 완료!")

        except Exception as e:
            print(f"❌ 메타데이터 로드 실패: {e}")

    def _categorize_mood_tags(self):
        """분위기 클래스를 카테고리별로 자동 분류"""
        if not self.mood_classes:
            return

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

        print(f"📊 카테고리 분류 완료:")
        print(f"   • 분위기: {len(self.mood_tags)}개")
        print(f"   • 테마: {len(self.theme_tags)}개")
        print(f"   • 기능: {len(self.function_tags)}개")

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
            },
            {
                "url": "https://essentia.upf.edu/models/music-style-classification/jamendo-genre-classes/genre_discogs400-discogs-effnet-1.pb",
                "path": self.genre_model_path,
                "name": "Discogs 400 Genre Classifier"
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

            # 장르 분류 모델 로드
            if self.genre_model_path.exists():
                print("🔄 장르 분류 모델 로드 중...")
                self.genre_graph, self.genre_session = self._load_pb_model(self.genre_model_path)
                if self.genre_session is not None:
                    print("✅ 장르 분류 모델 로드 완료!")

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

            # 태그별 딕셔너리 생성 (JSON에서 로드된 클래스 사용)
            activations_dict = {}
            mood_classes = self.mood_classes if self.mood_classes else []

            for i, tag in enumerate(mood_classes):
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
                        n_mels=128,  # 분석 결과에서 확인된 값
                        n_fft=2048,
                        hop_length=512,
                        fmin=0,
                        fmax=sr / 2
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

    def classify_genre(self, embeddings):
        """
        임베딩으로부터 400개 장르 분류 (장르명 포함)
        """
        if self.genre_session is None or embeddings is None:
            return None

        try:
            # 임베딩 평균내기 (배치 차원 제거)
            if len(embeddings.shape) > 1:
                embeddings_mean = np.mean(embeddings, axis=0, keepdims=True)  # [1, 1280]
            else:
                embeddings_mean = embeddings.reshape(1, -1)

            # 디버깅 정보 추가
            print(f"🔍 장르 분류 디버깅:")
            print(f"   임베딩 입력 형태: {embeddings_mean.shape}")
            print(f"   임베딩 범위 (정규화 전): [{embeddings_mean.min():.3f}, {embeddings_mean.max():.3f}]")

            # 여러 정규화 방식 시도
            # 방식 1: 기존 z-score 정규화
            embeddings_normalized = (embeddings_mean - embeddings_mean.mean()) / (embeddings_mean.std() + 1e-8)
            print(f"   z-score 정규화 후: [{embeddings_normalized.min():.3f}, {embeddings_normalized.max():.3f}]")

            # 방식 2: Min-Max 정규화 (0-1)
            embeddings_minmax = (embeddings_mean - embeddings_mean.min()) / (
                        embeddings_mean.max() - embeddings_mean.min() + 1e-8)
            print(f"   MinMax 정규화 후: [{embeddings_minmax.min():.3f}, {embeddings_minmax.max():.3f}]")

            # 방식 3: L2 정규화
            embeddings_l2 = embeddings_mean / (np.linalg.norm(embeddings_mean, axis=1, keepdims=True) + 1e-8)
            print(f"   L2 정규화 후: [{embeddings_l2.min():.3f}, {embeddings_l2.max():.3f}]")

            # 먼저 z-score 정규화로 시도
            final_embeddings = embeddings_normalized

            # 입력/출력 텐서 가져오기
            input_tensor = self.genre_graph.get_tensor_by_name(self.genre_input_name)
            output_tensor = self.genre_graph.get_tensor_by_name(self.genre_output_name)

            print(f"   입력 텐서: {self.genre_input_name}")
            print(f"   출력 텐서: {self.genre_output_name}")

            # 추론 실행
            predictions = self.genre_session.run(
                output_tensor,
                feed_dict={input_tensor: final_embeddings}  # 정규화된 임베딩 사용
            )

            # 장르 결과 처리 개선
            if len(predictions.shape) > 1:
                predictions = predictions[0]

            print(f"장르 예측 결과 형태: {predictions.shape}")
            print(f"장르 예측 범위: [{predictions.min():.3f}, {predictions.max():.3f}]")
            print(f"1.0인 값의 개수: {np.sum(predictions == 1.0)}")
            print(f"0.999 이상인 값의 개수: {np.sum(predictions >= 0.999)}")

            # 상위 10개 장르만 반환 (400개는 너무 많음)
            top_indices = np.argsort(predictions)[-10:][::-1]  # 상위 10개, 내림차순

            genre_results = []
            for idx in top_indices:
                # 0.999 미만인 값들만 의미있는 결과로 간주 (더 엄격하게)
                if predictions[idx] < 0.999:
                    genre_name = self.genre_classes[idx] if idx < len(self.genre_classes) else f"Unknown Genre {idx}"
                    genre_results.append({
                        'index': int(idx),
                        'genre': genre_name,
                        'score': float(predictions[idx])
                    })

            # 만약 모든 값이 0.999에 가깝다면, 상위 5개만 반환 (디버깅용)
            if len(genre_results) == 0:
                print("⚠️ 모든 값이 0.999 이상 - 상위 5개만 표시 (디버깅용)")
                for idx in top_indices[:5]:
                    genre_name = self.genre_classes[idx] if idx < len(self.genre_classes) else f"Unknown Genre {idx}"
                    genre_results.append({
                        'index': int(idx),
                        'genre': genre_name,
                        'score': float(predictions[idx])
                    })

            return genre_results

        except Exception as e:
            print(f"장르 분류 실패: {e}")
            print(f"입력 텐서: {self.genre_input_name}")
            print(f"출력 텐서: {self.genre_output_name}")
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
            print(f"✅ 오디오 로드 완료: {len(audio) / sr:.1f}초")
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

        # 5. 장르 분류
        print("🔄 장르 분류 중...")
        genre_results = self.classify_genre(embeddings)
        if genre_results is None:
            print("⚠️ 장르 분류 실패, 분위기 분석만 진행합니다.")

        # 6. 결과 처리
        try:
            print("🔄 결과 처리 중...")

            # 아웃라이어 임계값 계산 (더 유연하게 조정)
            values = list(activations.values())
            q1 = np.quantile(values, 0.25)
            q3 = np.quantile(values, 0.75)
            iqr = q3 - q1

            # 임계값을 더 관대하게 설정
            if iqr > 0:
                outlier_threshold = q3 + (1.0 * iqr)  # 1.5 대신 1.0 사용
            else:
                # IQR이 0이면 median 기준으로 설정
                median_val = np.median(values)
                outlier_threshold = median_val + (0.1 * median_val)

            print(f"임계값: {outlier_threshold:.3f}")
            print(f"Q1: {q1:.3f}, Q3: {q3:.3f}, IQR: {iqr:.3f}")

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

            result = {
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

            # 장르 결과 추가 (있는 경우)
            if genre_results is not None:
                result["genres"] = genre_results
                result["top_genres"] = genre_results[:5]  # 상위 5개만

            return result

        except Exception as e:
            return {"error": f"결과 처리 실패: {e}"}

    def close_sessions(self):
        """TensorFlow 세션들을 정리"""
        if self.embeddings_session:
            self.embeddings_session.close()
        if self.mood_session:
            self.mood_session.close()
        if self.genre_session:
            self.genre_session.close()
        print("✅ TensorFlow 세션들이 정리되었습니다.")


def main():
    """
    메인 실행 함수
    """
    classifier = CorrectedMusicMoodClassifier()

    # 음악 파일 경로
    music_file = "./musics/Pierce_Murphy_-_A_Serpent_I_Did_Hear.mp3"  # 실제 파일 경로로 변경

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

            # 장르 결과 출력 개선 (실제 장르명 포함)
            if 'top_genres' in result and result['top_genres']:
                print(f"\n🎼 상위 장르:")
                for genre_info in result['top_genres']:
                    print(f"  • {genre_info['genre']}: {genre_info['score']:.3f}")
            else:
                print(f"\n🎼 상위 장르:")
                print("  (의미있는 장르 결과 없음 - 모든 값이 1.0에 가까움)")

            print(f"\n📊 상위 태그 (신뢰도순):")
            for tag, score in result['top_tags']:
                print(f"  • {tag}: {score:.3f}")

    finally:
        # 세션 정리
        classifier.close_sessions()


if __name__ == "__main__":
    print("🎵 === JSON 메타데이터 기반 음악 분류기 ===")
    print("모든 클래스 정보를 JSON에서 동적으로 로드")
    print()

    try:
        main()
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다.")