"""
최종 음악 분위기 분류기
Essentia 완전 재현 전처리 사용
"""

import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
import requests
import json
import warnings

warnings.filterwarnings('ignore')


def preprocess_essentia_exact(audio, sr=16000, debug=False):
    """
    Essentia TensorflowInputMusiCNN 완전 재현
    shift=1, scale=10000, log10 적용
    """
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    if debug:
        print(f"🔍 [전처리] 오디오 길이: {len(audio)/sr:.2f}초")

    # Essentia 정확한 파라미터
    frame_size = 512
    hop_size = 256
    n_mels = 96
    patch_size = 128

    # 멜 스펙트로그램 (파워 스케일, 선형)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=frame_size,
        hop_length=hop_size,
        fmin=0.0,
        fmax=8000.0,
        power=2.0,  # 파워 스펙트로그램
        norm='slaney',
        htk=False
    )

    # Essentia 방식 로그 압축 (핵심!)
    mel_bands_shifted = (mel_spec + 1) * 10000
    mel_bands_log = np.log10(mel_bands_shifted)

    if debug:
        print(f"🔍 [전처리] 멜 스펙트로그램 형태: {mel_bands_log.shape}")
        print(f"🔍 [전처리] 값 범위: [{mel_bands_log.min():.3f}, {mel_bands_log.max():.3f}]")
        print(f"🔍 [전처리] 평균: {mel_bands_log.mean():.3f}")

    # 패치 생성
    n_frames = mel_bands_log.shape[1]
    patches = []
    patch_hop_size = 64

    for start in range(0, n_frames - patch_size + 1, patch_hop_size):
        end = start + patch_size
        patch = mel_bands_log[:, start:end]
        patches.append(patch)

    if len(patches) == 0 or n_frames >= patch_size:
        if n_frames >= patch_size:
            last_patch = mel_bands_log[:, -patch_size:]
            if len(patches) == 0 or not np.array_equal(patches[-1], last_patch):
                patches.append(last_patch)

    # 64개 패치로 조정
    if len(patches) > 64:
        indices = np.linspace(0, len(patches) - 1, 64, dtype=int)
        patches = [patches[i] for i in indices]
    elif len(patches) < 64:
        while len(patches) < 64:
            patches.append(patches[-1].copy())

    # 배치로 결합: [64, 96, 128] -> [64, 128, 96]
    mel_batch = np.array(patches, dtype=np.float32)
    mel_batch = np.transpose(mel_batch, (0, 2, 1))

    if debug:
        print(f"🔍 [전처리] 최종 배치 형태: {mel_batch.shape}")
        print(f"🔍 [전처리] 최종 범위: [{mel_batch.min():.3f}, {mel_batch.max():.3f}]")

    return mel_batch


class FinalMusicMoodClassifier:
    def __init__(self, model_dir="models", debug=True):
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

        # 텐서 이름
        self.embeddings_input_name = "serving_default_melspectrogram:0"
        self.embeddings_output_embeddings = "PartitionedCall:1"
        self.embeddings_output_direct_genre = "PartitionedCall:0"

        self.mood_input_name = "model/Placeholder:0"
        self.mood_output_name = "model/Sigmoid:0"

        self.genre_input_name = "serving_default_model_Placeholder:0"
        self.genre_output_name = "PartitionedCall:0"

        # 클래스 리스트들
        self.genre_classes = []
        self.mood_classes = []

        # TensorFlow 세션들
        self.embeddings_session = None
        self.embeddings_graph = None
        self.mood_session = None
        self.mood_graph = None
        self.genre_session = None
        self.genre_graph = None

        # 초기화
        self._download_metadata()
        self._load_metadata()
        self._download_models()
        self._load_tensorflow_models()

    def _debug_print(self, message):
        if self.debug:
            print(f"🔍 DEBUG: {message}")

    def _download_metadata(self):
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

        except Exception as e:
            print(f"❌ 메타데이터 로드 실패: {e}")

    def _download_models(self):
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

    def predict_final(self, file_path):
        """
        최종 예측 - Essentia 완전 재현 전처리 사용
        """
        print(f"🎵 최종 분석 시작: {file_path}")
        print("Essentia 완전 재현 전처리 사용 (shift=1, scale=10000, log10)")
        print("=" * 70)

        # 오디오 로드
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            print(f"✅ 오디오 로드 완료: {len(audio) / sr:.1f}초")
        except Exception as e:
            return {"error": f"오디오 로드 실패: {e}"}

        # Essentia 완전 재현 전처리
        print(f"\n🔄 Essentia 완전 재현 전처리")
        mel_spectrogram = preprocess_essentia_exact(audio, sr, debug=self.debug)

        if mel_spectrogram is None:
            return {"error": "전처리 실패"}

        # 임베딩 및 직접 예측 추출
        print(f"\n🔄 임베딩 및 예측 추출")
        embeddings, direct_predictions = self.extract_embeddings_and_direct_predictions(mel_spectrogram)

        if embeddings is None:
            return {"error": "임베딩 추출 실패"}

        # 장르 분류
        print(f"\n🔄 장르 분류")
        genre_results = self.classify_genre_all_methods(embeddings, direct_predictions)

        # 분위기 분류
        print(f"\n🔄 분위기 분류")
        mood_results = self.classify_mood(embeddings)

        # 결과 정리
        result = {
            "file_path": file_path,
            "preprocessing_method": "essentia_exact",
            "mel_spectrogram_stats": {
                "shape": mel_spectrogram.shape,
                "min": float(mel_spectrogram.min()),
                "max": float(mel_spectrogram.max()),
                "mean": float(mel_spectrogram.mean()),
                "std": float(mel_spectrogram.std())
            },
            "embeddings_stats": {
                "shape": embeddings.shape,
                "min": float(embeddings.min()),
                "max": float(embeddings.max()),
                "mean": float(embeddings.mean()),
                "std": float(embeddings.std())
            },
            "genre_results": genre_results,
            "mood_results": mood_results,
            "model_used": "Essentia Discogs-EfficientNet (완전 재현)"
        }

        return result

    def extract_embeddings_and_direct_predictions(self, mel_spectrogram):
        """임베딩과 직접 예측 모두 추출"""
        if self.embeddings_session is None or mel_spectrogram is None:
            return None, None

        try:
            input_tensor = self.embeddings_graph.get_tensor_by_name(self.embeddings_input_name)
            embeddings_tensor = self.embeddings_graph.get_tensor_by_name(self.embeddings_output_embeddings)
            direct_predictions_tensor = self.embeddings_graph.get_tensor_by_name(self.embeddings_output_direct_genre)

            embeddings, direct_predictions = self.embeddings_session.run(
                [embeddings_tensor, direct_predictions_tensor],
                feed_dict={input_tensor: mel_spectrogram}
            )

            self._debug_print(f"임베딩 형태: {embeddings.shape}")
            self._debug_print(f"임베딩 범위: [{embeddings.min():.6f}, {embeddings.max():.6f}]")
            self._debug_print(f"직접 예측 형태: {direct_predictions.shape}")
            self._debug_print(f"직접 예측 범위: [{direct_predictions.min():.6f}, {direct_predictions.max():.6f}]")

            return embeddings, direct_predictions

        except Exception as e:
            print(f"임베딩/예측 추출 실패: {e}")
            return None, None

    def classify_genre_all_methods(self, embeddings, direct_predictions=None):
        """모든 방법으로 장르 분류"""
        results = {}

        # 방법 1: 직접 예측 사용
        if direct_predictions is not None:
            try:
                avg_direct_predictions = np.mean(direct_predictions, axis=0)
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
                self._debug_print(f"직접 예측 최고 점수: {avg_direct_predictions.max():.6f}")

            except Exception as e:
                print(f"직접 예측 처리 실패: {e}")

        # 방법 2: 임베딩을 통한 장르 분류
        if embeddings is not None and self.genre_session is not None:
            try:
                # 패치별 예측
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

                # 집계 방법들
                mean_predictions = np.mean(all_predictions, axis=0)
                max_predictions = np.max(all_predictions, axis=0)

                for method_name, predictions in [
                    ('embedding_mean', mean_predictions),
                    ('embedding_max', max_predictions)
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
                    self._debug_print(f"{method_name} 최고 점수: {predictions.max():.6f}")

            except Exception as e:
                print(f"임베딩 기반 장르 분류 실패: {e}")

        return results

    def classify_mood(self, embeddings):
        """분위기 분류"""
        if self.mood_session is None or embeddings is None:
            return None

        try:
            # 임베딩 평균
            embeddings_mean = np.mean(embeddings, axis=0, keepdims=True)

            input_tensor = self.mood_graph.get_tensor_by_name(self.mood_input_name)
            output_tensor = self.mood_graph.get_tensor_by_name(self.mood_output_name)

            predictions = self.mood_session.run(
                output_tensor,
                feed_dict={input_tensor: embeddings_mean}
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

    def analyze_final_results(self, result):
        """최종 결과 분석"""
        print(f"\n{'='*70}")
        print("📊 최종 결과 분석")
        print(f"{'='*70}")

        if "error" in result:
            print(f"❌ 오류: {result['error']}")
            return

        print(f"📁 파일: {result['file_path']}")
        print(f"🔬 전처리: {result['preprocessing_method']}")

        # 전처리 통계
        mel_stats = result['mel_spectrogram_stats']
        print(f"\n📊 멜 스펙트로그램 통계:")
        print(f"   형태: {mel_stats['shape']}")
        print(f"   범위: [{mel_stats['min']:.3f}, {mel_stats['max']:.3f}]")
        print(f"   평균: {mel_stats['mean']:.3f}")

        # 임베딩 통계
        emb_stats = result['embeddings_stats']
        print(f"\n🧠 임베딩 통계:")
        print(f"   형태: {emb_stats['shape']}")
        print(f"   범위: [{emb_stats['min']:.6f}, {emb_stats['max']:.6f}]")
        print(f"   평균: {emb_stats['mean']:.6f}")

        # 임베딩 범위 확인
        if -1 <= emb_stats['min'] and emb_stats['max'] <= 1:
            print("   ✅ 임베딩 범위가 정상적입니다 [-1, 1]")
        else:
            print(f"   ❌ 임베딩 범위가 비정상적입니다")

        # 장르 결과
        genre_results = result.get('genre_results', {})
        print(f"\n🎼 장르 분류 결과:")

        all_genre_votes = {}
        for method_name, genre_list in genre_results.items():
            if genre_list and len(genre_list) > 0:
                top_genre = genre_list[0]
                genre_name = top_genre['genre']
                score = top_genre['score']

                print(f"   • {method_name}: {genre_name} ({score:.6f})")

                if genre_name not in all_genre_votes:
                    all_genre_votes[genre_name] = []
                all_genre_votes[genre_name].append(score)

        # 최종 추천 장르
        if all_genre_votes:
            # 평균 점수로 정렬
            genre_avg_scores = {genre: np.mean(scores) for genre, scores in all_genre_votes.items()}
            best_genre = max(genre_avg_scores, key=genre_avg_scores.get)
            best_score = genre_avg_scores[best_genre]
            vote_count = len(all_genre_votes[best_genre])

            print(f"\n🎯 최종 추천 장르: {best_genre}")
            print(f"   투표 수: {vote_count}")
            print(f"   평균 점수: {best_score:.6f}")

            # Classical 관련 체크
            # if 'classical' in best_genre.lower() or 'baroque' in best_genre.lower() or 'romantic' in best_genre.lower():
            #     print("   ✅ 클래식 음악으로 올바르게 분류되었습니다!")
            # else:
            #     print("   ❌ 여전히 클래식으로 분류되지 않았습니다.")

        # 분위기 결과 (상위 10개만)
        mood_results = result.get('mood_results')
        if mood_results:
            print(f"\n🎭 분위기 분류 결과 (상위 10개):")
            sorted_moods = sorted(mood_results.items(), key=lambda x: x[1], reverse=True)[:10]
            for tag, score in sorted_moods:
                print(f"   • {tag}: {score:.6f}")

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
    """최종 메인 실행 함수"""
    classifier = FinalMusicMoodClassifier(debug=True)

    # 음악 파일 경로
    music_file = "./musics/Anitek_-_Tab_+_Anitek_-_Bleach.mp3"

    print(f"🎵 === 최종 음악 분류기 (Essentia 완전 재현) ===")
    print(f"shift=1, scale=10000, log10 적용")
    print(f"📁 파일: {music_file}")
    print("=" * 70)

    try:
        # 최종 예측
        result = classifier.predict_final(music_file)

        # 결과 분석
        classifier.analyze_final_results(result)

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()

    finally:
        classifier.close_sessions()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다.")