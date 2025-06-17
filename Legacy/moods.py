import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
import requests
import warnings
import json
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class ImprovedMusicMoodClassifier:
    def __init__(self, model_dir="models", debug=True):
        """
        개선된 음악 분위기 분류기 - 전처리와 모델 사용 방식을 대폭 개선
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.debug = debug

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
        self.embeddings_output_embeddings = "PartitionedCall:1"  # [64, 1280] 임베딩
        self.embeddings_output_direct_genre = "PartitionedCall:0"  # [64, 400] 직접 장르?
        self.embeddings_input_shape = [64, 128, 96]  # [batch, time_frames, mel_bins]

        self.mood_input_name = "model/Placeholder:0"
        self.mood_output_name = "model/Sigmoid:0"

        # 분석 결과에 따른 정확한 장르 모델 텐서
        self.genre_input_name = "serving_default_model_Placeholder:0"
        self.genre_output_name = "PartitionedCall:0"

        # 클래스 리스트들
        self.genre_classes = []
        self.mood_classes = []
        self.embeddings_classes = []

        # 카테고리별 분류
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

        # 전처리 파라미터 (Essentia 기본값에 더 가깝게)
        self.mel_params = {
            'sr': 16000,
            'n_fft': 512,
            'hop_length': 256,
            'n_mels': 96,
            'fmin': 0.0,
            'fmax': 8000.0,  # sr/2
            'power': 2.0,
            'patch_size': 128,
            'patch_hop_size': 62,
            'target_patches': 64
        }

        # 초기화
        self._download_metadata()
        self._load_metadata()
        self._download_models()
        self._load_tensorflow_models()

    def _debug_print(self, message):
        """디버그 메시지 출력"""
        if self.debug:
            print(f"🔍 DEBUG: {message}")

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
            if self.genre_json_path.exists():
                with open(self.genre_json_path, 'r', encoding='utf-8') as f:
                    genre_metadata = json.load(f)
                    self.genre_classes = genre_metadata.get('classes', [])
                    print(f"✅ 장르 클래스 {len(self.genre_classes)}개 로드 완료!")

            if self.mood_json_path.exists():
                with open(self.mood_json_path, 'r', encoding='utf-8') as f:
                    mood_metadata = json.load(f)
                    self.mood_classes = mood_metadata.get('classes', [])
                    print(f"✅ 분위기 클래스 {len(self.mood_classes)}개 로드 완료!")
                    self._categorize_mood_tags()

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

        mood_keywords = {
            'calm', 'cool', 'dark', 'deep', 'dramatic', 'emotional', 'energetic',
            'epic', 'fast', 'fun', 'funny', 'groovy', 'happy', 'heavy', 'hopeful',
            'inspiring', 'meditative', 'melancholic', 'motivational', 'positive',
            'powerful', 'relaxing', 'romantic', 'sad', 'sexy', 'slow', 'soft',
            'upbeat', 'uplifting'
        }

        theme_keywords = {
            'action', 'adventure', 'ballad', 'children', 'christmas', 'dream',
            'film', 'game', 'holiday', 'love', 'movie', 'nature', 'party',
            'retro', 'space', 'sport', 'summer', 'travel'
        }

        function_keywords = {
            'advertising', 'background', 'commercial', 'corporate', 'documentary',
            'drama', 'soundscape', 'trailer'
        }

        self.mood_tags = [tag for tag in self.mood_classes if tag in mood_keywords]
        self.theme_tags = [tag for tag in self.mood_classes if tag in theme_keywords]
        self.function_tags = [tag for tag in self.mood_classes if tag in function_keywords]

        print(f"📊 카테고리 분류 완료: 분위기 {len(self.mood_tags)}개, 테마 {len(self.theme_tags)}개, 기능 {len(self.function_tags)}개")

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
            if self.embeddings_model_path.exists():
                print("🔄 임베딩 모델 로드 중...")
                self.embeddings_graph, self.embeddings_session = self._load_pb_model(self.embeddings_model_path)
                if self.embeddings_session is not None:
                    print("✅ 임베딩 모델 로드 완료!")

            if self.mood_model_path.exists():
                print("🔄 분위기 분류 모델 로드 중...")
                self.mood_graph, self.mood_session = self._load_pb_model(self.mood_model_path)
                if self.mood_session is not None:
                    print("✅ 분위기 분류 모델 로드 완료!")

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
            with tf.io.gfile.GFile(str(model_path), "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(graph_def, name="")

            session = tf.compat.v1.Session(graph=graph)
            return graph, session

        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return None, None

    def preprocess_audio_improved(self, audio, sr=16000, method="essentia_accurate"):
        """
        개선된 오디오 전처리 - 여러 방법 지원
        """
        try:
            # 1. 리샘플링
            if sr != self.mel_params['sr']:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.mel_params['sr'])
                sr = self.mel_params['sr']

            self._debug_print(f"오디오 길이: {len(audio) / sr:.2f}초")

            if method == "essentia_accurate":
                return self._preprocess_essentia_style(audio, sr)
            elif method == "librosa_improved":
                return self._preprocess_librosa_improved(audio, sr)
            elif method == "original":
                return self._preprocess_original(audio, sr)
            else:
                raise ValueError(f"알 수 없는 전처리 방법: {method}")

        except Exception as e:
            print(f"오디오 전처리 실패: {e}")
            return None

    def _preprocess_essentia_style(self, audio, sr):
        """Essentia 스타일 전처리 (더 정확한 파라미터)"""
        # Essentia에서 사용하는 정확한 파라미터
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.mel_params['n_mels'],
            n_fft=self.mel_params['n_fft'],
            hop_length=self.mel_params['hop_length'],
            fmin=self.mel_params['fmin'],
            fmax=self.mel_params['fmax'],
            power=self.mel_params['power'],
            norm='slaney',  # Essentia 스타일
            htk=False
        )

        # dB 변환 (Essentia 방식)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, amin=1e-10, top_db=80.0)

        # 정규화 (0-1 범위로)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

        self._debug_print(f"멜 스펙트로그램 형태: {mel_spec_norm.shape}")
        self._debug_print(f"값 범위: [{mel_spec_norm.min():.3f}, {mel_spec_norm.max():.3f}]")

        return self._create_patches_smart(mel_spec_norm)

    def _preprocess_librosa_improved(self, audio, sr):
        """개선된 librosa 전처리"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.mel_params['n_mels'],
            n_fft=self.mel_params['n_fft'],
            hop_length=self.mel_params['hop_length'],
            fmin=self.mel_params['fmin'],
            fmax=self.mel_params['fmax'],
            power=self.mel_params['power']
        )

        # dB 변환
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 표준화 (평균 0, 표준편차 1)
        mel_spec_std = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()

        return self._create_patches_smart(mel_spec_std)

    def _preprocess_original(self, audio, sr):
        """원래 방식 (비교용)"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.mel_params['n_mels'],
            n_fft=self.mel_params['n_fft'],
            hop_length=self.mel_params['hop_length'],
            fmin=self.mel_params['fmin'],
            fmax=self.mel_params['fmax']
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return self._create_patches_original(mel_spec_db)

    def _create_patches_smart(self, mel_spec):
        """스마트 패치 생성 - 여러 전략 지원"""
        n_frames = mel_spec.shape[1]
        patch_size = self.mel_params['patch_size']
        target_patches = self.mel_params['target_patches']

        self._debug_print(f"전체 프레임 수: {n_frames}, 패치 크기: {patch_size}")

        patches = []

        if n_frames < patch_size:
            # 너무 짧은 경우: 패딩
            padded_spec = np.pad(mel_spec, ((0, 0), (0, patch_size - n_frames)), mode='constant')
            patches.append(padded_spec)
        else:
            # 충분한 길이: 여러 전략으로 패치 생성

            # 1. 균등 샘플링
            if n_frames >= patch_size * 2:
                step = max(1, (n_frames - patch_size) // (target_patches - 1))
                for i in range(0, min(n_frames - patch_size + 1, target_patches * step), step):
                    patch = mel_spec[:, i:i + patch_size]
                    patches.append(patch)

            # 2. 중요 구간 추가 (에너지가 높은 구간)
            if len(patches) < target_patches:
                # 프레임별 에너지 계산
                frame_energy = np.mean(mel_spec ** 2, axis=0)

                # 에너지가 높은 구간 찾기
                for i in range(min(target_patches - len(patches), 10)):
                    # 아직 사용하지 않은 구간에서 에너지가 높은 곳 찾기
                    start = i * (n_frames // 10)
                    end = min(start + patch_size, n_frames)
                    if end - start >= patch_size:
                        patch = mel_spec[:, start:start + patch_size]
                        patches.append(patch)

        # 패치 수 조정
        if len(patches) > target_patches:
            # 에너지 기준으로 상위 패치 선택
            energies = [np.mean(patch ** 2) for patch in patches]
            top_indices = np.argsort(energies)[-target_patches:]
            patches = [patches[i] for i in sorted(top_indices)]
        elif len(patches) < target_patches:
            # 부족한 경우 마지막 패치 반복
            while len(patches) < target_patches:
                patches.append(patches[-1].copy())

        # 배치로 결합: [64, 96, 128] -> [64, 128, 96]
        mel_batch = np.array(patches, dtype=np.float32)
        mel_batch = np.transpose(mel_batch, (0, 2, 1))

        self._debug_print(f"최종 배치 형태: {mel_batch.shape}")
        self._debug_print(f"배치 값 범위: [{mel_batch.min():.3f}, {mel_batch.max():.3f}]")

        return mel_batch

    def _create_patches_original(self, mel_spec_db):
        """원래 패치 생성 방식"""
        n_frames = mel_spec_db.shape[1]
        patches = []
        patch_starts = list(range(0, n_frames - self.mel_params['patch_size'] + 1, self.mel_params['patch_hop_size']))

        if patch_starts[-1] + self.mel_params['patch_size'] < n_frames:
            patch_starts.append(n_frames - self.mel_params['patch_size'])

        for start in patch_starts:
            end = start + self.mel_params['patch_size']
            patch = mel_spec_db[:, start:end]
            patches.append(patch)

        # 64개로 맞추기
        if len(patches) > 64:
            indices = np.linspace(0, len(patches) - 1, 64, dtype=int)
            patches = [patches[i] for i in indices]
        elif len(patches) < 64:
            while len(patches) < 64:
                patches.append(patches[-1])

        mel_batch = np.array(patches, dtype=np.float32)
        mel_batch = np.transpose(mel_batch, (0, 2, 1))
        return mel_batch

    def extract_embeddings_and_direct_predictions(self, mel_spectrogram):
        """
        임베딩과 직접 예측 모두 추출
        """
        if self.embeddings_session is None or mel_spectrogram is None:
            return None, None

        try:
            input_tensor = self.embeddings_graph.get_tensor_by_name(self.embeddings_input_name)
            embeddings_tensor = self.embeddings_graph.get_tensor_by_name(self.embeddings_output_embeddings)
            direct_predictions_tensor = self.embeddings_graph.get_tensor_by_name(self.embeddings_output_direct_genre)

            # 두 출력 모두 추출
            embeddings, direct_predictions = self.embeddings_session.run(
                [embeddings_tensor, direct_predictions_tensor],
                feed_dict={input_tensor: mel_spectrogram}
            )

            self._debug_print(f"임베딩 형태: {embeddings.shape}")
            self._debug_print(f"직접 예측 형태: {direct_predictions.shape}")

            return embeddings, direct_predictions

        except Exception as e:
            print(f"임베딩/예측 추출 실패: {e}")
            return None, None

    def classify_genre_multiple_methods(self, embeddings, direct_predictions=None):
        """
        여러 방법으로 장르 분류
        """
        results = {}

        # 방법 1: 직접 예측 사용 (만약 있다면)
        if direct_predictions is not None:
            try:
                # 패치별 예측을 평균
                avg_direct_predictions = np.mean(direct_predictions, axis=0)

                # 상위 장르 추출
                top_indices = np.argsort(avg_direct_predictions)[-10:][::-1]
                direct_results = []
                for idx in top_indices:
                    genre_name = self.genre_classes[idx] if idx < len(self.genre_classes) else f"Unknown {idx}"
                    direct_results.append({
                        'index': int(idx),
                        'genre': genre_name,
                        'score': float(avg_direct_predictions[idx])
                    })

                results['direct_predictions'] = direct_results
                self._debug_print(f"직접 예측 최고 점수: {avg_direct_predictions.max():.4f}")

            except Exception as e:
                print(f"직접 예측 처리 실패: {e}")

        # 방법 2: 임베딩을 통한 장르 분류 (기존 방식)
        if embeddings is not None and self.genre_session is not None:
            try:
                # 각 패치별로 예측 후 집계
                all_predictions = []
                for i in range(embeddings.shape[0]):
                    patch_embedding = embeddings[i:i + 1]

                    input_tensor = self.genre_graph.get_tensor_by_name(self.genre_input_name)
                    output_tensor = self.genre_graph.get_tensor_by_name(self.genre_output_name)

                    prediction = self.genre_session.run(
                        output_tensor,
                        feed_dict={input_tensor: patch_embedding}
                    )

                    if len(prediction.shape) > 1:
                        prediction = prediction[0]

                    all_predictions.append(prediction)

                # 여러 집계 방법 시도
                mean_predictions = np.mean(all_predictions, axis=0)
                max_predictions = np.max(all_predictions, axis=0)
                median_predictions = np.median(all_predictions, axis=0)

                # 각 방법별 결과
                for method_name, predictions in [
                    ('embedding_mean', mean_predictions),
                    ('embedding_max', max_predictions),
                    ('embedding_median', median_predictions)
                ]:
                    top_indices = np.argsort(predictions)[-10:][::-1]
                    method_results = []
                    for idx in top_indices:
                        genre_name = self.genre_classes[idx] if idx < len(self.genre_classes) else f"Unknown {idx}"
                        method_results.append({
                            'index': int(idx),
                            'genre': genre_name,
                            'score': float(predictions[idx])
                        })

                    results[method_name] = method_results
                    self._debug_print(f"{method_name} 최고 점수: {predictions.max():.4f}")

            except Exception as e:
                print(f"임베딩 기반 장르 분류 실패: {e}")

        return results

    def classify_mood_improved(self, embeddings, aggregation_method="mean"):
        """
        개선된 분위기 분류
        """
        if self.mood_session is None or embeddings is None:
            return None

        try:
            # 여러 집계 방법
            if aggregation_method == "mean":
                embeddings_agg = np.mean(embeddings, axis=0, keepdims=True)
            elif aggregation_method == "max":
                embeddings_agg = np.max(embeddings, axis=0, keepdims=True)
            elif aggregation_method == "weighted":
                # 에너지 기반 가중 평균
                weights = np.mean(embeddings ** 2, axis=1)
                weights = weights / weights.sum()
                embeddings_agg = np.average(embeddings, axis=0, weights=weights, keepdims=True)
            else:
                embeddings_agg = np.mean(embeddings, axis=0, keepdims=True)

            # 정규화 제거하고 원본 사용
            input_tensor = self.mood_graph.get_tensor_by_name(self.mood_input_name)
            output_tensor = self.mood_graph.get_tensor_by_name(self.mood_output_name)

            predictions = self.mood_session.run(
                output_tensor,
                feed_dict={input_tensor: embeddings_agg}
            )

            if len(predictions.shape) > 1:
                predictions = predictions[0]

            # 태그별 딕셔너리 생성
            activations_dict = {}
            for i, tag in enumerate(self.mood_classes):
                if i < len(predictions):
                    activations_dict[tag] = float(predictions[i])

            return activations_dict

        except Exception as e:
            print(f"분위기 분류 실패: {e}")
            return None

    def predict_comprehensive(self, file_path, preprocessing_methods=None, aggregation_methods=None):
        """
        포괄적인 예측 - 여러 방법으로 시도하고 비교
        """
        if preprocessing_methods is None:
            preprocessing_methods = ["essentia_accurate", "librosa_improved", "original"]

        if aggregation_methods is None:
            aggregation_methods = ["mean", "max", "weighted"]

        print(f"🎵 포괄적 분석 시작: {file_path}")
        print("=" * 70)

        # 오디오 로드
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            print(f"✅ 오디오 로드 완료: {len(audio) / sr:.1f}초")
        except Exception as e:
            return {"error": f"오디오 로드 실패: {e}"}

        all_results = {}

        # 각 전처리 방법별로 시도
        for prep_method in preprocessing_methods:
            print(f"\n🔄 전처리 방법: {prep_method}")

            # 전처리
            mel_spectrogram = self.preprocess_audio_improved(audio, sr, method=prep_method)
            if mel_spectrogram is None:
                print(f"❌ {prep_method} 전처리 실패")
                continue

            # 임베딩 및 직접 예측 추출
            embeddings, direct_predictions = self.extract_embeddings_and_direct_predictions(mel_spectrogram)
            if embeddings is None:
                print(f"❌ {prep_method} 임베딩 추출 실패")
                continue

            # 장르 분류 (여러 방법)
            genre_results = self.classify_genre_multiple_methods(embeddings, direct_predictions)

            # 분위기 분류 (여러 집계 방법)
            mood_results = {}
            for agg_method in aggregation_methods:
                mood_result = self.classify_mood_improved(embeddings, agg_method)
                if mood_result is not None:
                    mood_results[agg_method] = mood_result

            # 결과 저장
            all_results[prep_method] = {
                'genre_results': genre_results,
                'mood_results': mood_results,
                'mel_spectrogram_shape': mel_spectrogram.shape,
                'embeddings_shape': embeddings.shape if embeddings is not None else None,
                'direct_predictions_shape': direct_predictions.shape if direct_predictions is not None else None
            }

            # 각 방법별 최고 장르 출력
            print(f"📊 {prep_method} 결과:")
            for method_name, results in genre_results.items():
                if results and len(results) > 0:
                    top_genre = results[0]
                    print(f"   • {method_name}: {top_genre['genre']} ({top_genre['score']:.4f})")

        return all_results

    def analyze_results(self, results):
        """
        결과 분석 및 최적 방법 추천
        """
        print(f"\n{'=' * 70}")
        print("📊 결과 종합 분석")
        print(f"{'=' * 70}")

        genre_votes = {}
        method_scores = {}

        # 각 방법별 장르 투표 집계
        for prep_method, prep_results in results.items():
            print(f"\n🔍 {prep_method} 상세 결과:")

            for method_name, genre_list in prep_results['genre_results'].items():
                if genre_list and len(genre_list) > 0:
                    top_genre = genre_list[0]
                    genre_name = top_genre['genre']
                    score = top_genre['score']

                    # 투표 집계
                    if genre_name not in genre_votes:
                        genre_votes[genre_name] = []
                    genre_votes[genre_name].append({
                        'method': f"{prep_method}_{method_name}",
                        'score': score
                    })

                    # 방법별 점수 기록
                    method_key = f"{prep_method}_{method_name}"
                    method_scores[method_key] = score

                    print(f"   • {method_name}: {genre_name} ({score:.4f})")

        # 투표 결과 분석
        print(f"\n🗳️  장르별 투표 결과:")
        for genre, votes in sorted(genre_votes.items(), key=lambda x: len(x[1]), reverse=True):
            avg_score = np.mean([vote['score'] for vote in votes])
            print(f"   • {genre}: {len(votes)}표, 평균 점수: {avg_score:.4f}")
            for vote in votes:
                print(f"     - {vote['method']}: {vote['score']:.4f}")

        # 최고 성능 방법 찾기
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            print(f"\n🏆 최고 성능 방법: {best_method} (점수: {best_score:.4f})")

        # 가장 많이 투표받은 장르
        if genre_votes:
            most_voted_genre = max(genre_votes, key=lambda x: len(genre_votes[x]))
            vote_count = len(genre_votes[most_voted_genre])
            avg_score = np.mean([vote['score'] for vote in genre_votes[most_voted_genre]])
            print(f"🎯 최종 추천 장르: {most_voted_genre} ({vote_count}표, 평균: {avg_score:.4f})")

        return {
            'genre_votes': genre_votes,
            'method_scores': method_scores,
            'best_method': best_method if method_scores else None,
            'recommended_genre': most_voted_genre if genre_votes else None
        }

    def visualize_spectrogram(self, file_path, method="essentia_accurate", save_path=None):
        """
        멜 스펙트로그램 시각화
        """
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            mel_spectrogram = self.preprocess_audio_improved(audio, sr, method=method)

            if mel_spectrogram is None:
                print("스펙트로그램 생성 실패")
                return

            # 첫 번째 패치만 시각화
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'멜 스펙트로그램 분석 - {method}', fontsize=16)

            # 첫 4개 패치 시각화
            for i in range(min(4, mel_spectrogram.shape[0])):
                row, col = i // 2, i % 2
                patch = mel_spectrogram[i].T  # [96, 128]로 전치

                im = axes[row, col].imshow(patch, aspect='auto', origin='lower', cmap='viridis')
                axes[row, col].set_title(f'패치 {i + 1}')
                axes[row, col].set_xlabel('시간 프레임')
                axes[row, col].set_ylabel('멜 주파수 빈')
                plt.colorbar(im, ax=axes[row, col])

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"스펙트로그램이 {save_path}에 저장되었습니다.")
            else:
                plt.show()

        except Exception as e:
            print(f"시각화 실패: {e}")

    def debug_model_outputs(self, file_path):
        """
        모델 출력을 상세히 디버깅
        """
        print(f"🔧 모델 출력 디버깅: {file_path}")
        print("=" * 60)

        try:
            audio, sr = librosa.load(file_path, sr=16000)
            mel_spec = self.preprocess_audio_improved(audio, sr, method="essentia_accurate")

            if mel_spec is None:
                print("전처리 실패")
                return

            # 임베딩 모델 디버깅
            embeddings, direct_preds = self.extract_embeddings_and_direct_predictions(mel_spec)

            if embeddings is not None:
                print(f"🔍 임베딩 통계:")
                print(f"   형태: {embeddings.shape}")
                print(f"   평균: {embeddings.mean():.6f}")
                print(f"   표준편차: {embeddings.std():.6f}")
                print(f"   범위: [{embeddings.min():.6f}, {embeddings.max():.6f}]")

            if direct_preds is not None:
                print(f"🔍 직접 예측 통계:")
                print(f"   형태: {direct_preds.shape}")
                print(f"   평균: {direct_preds.mean():.6f}")
                print(f"   표준편차: {direct_preds.std():.6f}")
                print(f"   범위: [{direct_preds.min():.6f}, {direct_preds.max():.6f}]")

                # 상위 예측 출력
                avg_preds = np.mean(direct_preds, axis=0)
                top_indices = np.argsort(avg_preds)[-5:][::-1]
                print(f"   상위 5개 직접 예측:")
                for idx in top_indices:
                    genre = self.genre_classes[idx] if idx < len(self.genre_classes) else f"Unknown_{idx}"
                    print(f"     {genre}: {avg_preds[idx]:.6f}")

            # 장르 모델 디버깅
            if embeddings is not None and self.genre_session is not None:
                print(f"🔍 장르 모델 출력:")

                # 첫 번째 패치로 테스트
                first_patch = embeddings[0:1]

                input_tensor = self.genre_graph.get_tensor_by_name(self.genre_input_name)
                output_tensor = self.genre_graph.get_tensor_by_name(self.genre_output_name)

                genre_output = self.genre_session.run(
                    output_tensor,
                    feed_dict={input_tensor: first_patch}
                )

                if len(genre_output.shape) > 1:
                    genre_output = genre_output[0]

                print(f"   형태: {genre_output.shape}")
                print(f"   평균: {genre_output.mean():.6f}")
                print(f"   표준편차: {genre_output.std():.6f}")
                print(f"   범위: [{genre_output.min():.6f}, {genre_output.max():.6f}]")

                # 상위 장르 출력
                top_indices = np.argsort(genre_output)[-5:][::-1]
                print(f"   상위 5개 장르 (첫 번째 패치):")
                for idx in top_indices:
                    genre = self.genre_classes[idx] if idx < len(self.genre_classes) else f"Unknown_{idx}"
                    print(f"     {genre}: {genre_output[idx]:.6f}")

        except Exception as e:
            print(f"디버깅 실패: {e}")

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
    개선된 메인 실행 함수
    """
    classifier = ImprovedMusicMoodClassifier(debug=True)

    # 음악 파일 경로
    music_file = "../musics/Debussy_-_Arabesque_-_Aufklarung.mp3"

    print(f"🎵 개선된 음악 분류기로 포괄적 분석")
    print(f"📁 파일: {music_file}")
    print("=" * 70)

    try:
        # 1. 모델 출력 디버깅
        print("\n🔧 1단계: 모델 출력 디버깅")
        classifier.debug_model_outputs(music_file)

        # 2. 포괄적 예측
        print(f"\n🔍 2단계: 포괄적 예측")
        results = classifier.predict_comprehensive(
            music_file,
            preprocessing_methods=["essentia_accurate", "librosa_improved"],
            aggregation_methods=["mean", "weighted"]
        )

        # 3. 결과 분석
        if results:
            analysis = classifier.analyze_results(results)

        # 4. 스펙트로그램 시각화 (선택사항)
        print(f"\n📊 3단계: 스펙트로그램 시각화")
        try:
            classifier.visualize_spectrogram(music_file, method="essentia_accurate", save_path="debug_spectrogram.png")
        except Exception as e:
            print(f"시각화 건너뛰기: {e}")

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()

    finally:
        classifier.close_sessions()


if __name__ == "__main__":
    print("🎵 === 개선된 음악 분류기 (포괄적 디버깅 버전) ===")
    print("여러 전처리 방법과 집계 방법을 시도하여 최적 결과 도출")
    print()

    try:
        main()
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback

        traceback.print_exc()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다.")