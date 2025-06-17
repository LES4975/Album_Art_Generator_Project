"""
음악 장르 및 분위기 분류기
사전학습된 Discogs EfficientNet과 MTG Jamendo 모델 사용
"""

import numpy as np
import librosa
import tensorflow as tf
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class MusicClassifier:
    def __init__(self, model_dir="models"):
        """
        음악 장르 및 분위기 분류기 초기화

        Args:
            model_dir: 모델 파일들이 저장된 디렉토리
        """
        self.model_dir = Path(model_dir)

        # 모델 파일 경로
        self.genre_model_path = self.model_dir / "discogs-effnet-bs64-1.pb"
        self.mood_model_path = self.model_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"

        # JSON 메타데이터 파일 경로
        self.genre_json_path = self.model_dir / "discogs-effnet-bs64-1.json"
        self.mood_json_path = self.model_dir / "mtg_jamendo_moodtheme-discogs-effnet-1.json"

        # 클래스 리스트
        self.genre_classes = []
        self.mood_classes = []

        # TensorFlow 세션과 그래프
        self.genre_session = None
        self.genre_graph = None
        self.mood_session = None
        self.mood_graph = None

        # 텐서 이름들 (JSON 메타데이터에서 확인)
        self.genre_input_name = "serving_default_melspectrogram:0"
        self.genre_output_predictions = "PartitionedCall:0"  # [64, 400] 장르 예측
        self.genre_output_embeddings = "PartitionedCall:1"  # [64, 1280] 임베딩

        self.mood_input_name = "model/Placeholder:0"
        self.mood_output_name = "model/Sigmoid:0"

        # 메타데이터 로드 및 모델 초기화
        self._load_metadata()
        self._load_models()

    def _load_metadata(self):
        """JSON 메타데이터에서 클래스 정보 로드"""
        try:
            # 장르 클래스 로드
            if self.genre_json_path.exists():
                with open(self.genre_json_path, 'r', encoding='utf-8') as f:
                    genre_metadata = json.load(f)
                    self.genre_classes = genre_metadata.get('classes', [])
                    print(f"✅ 장르 클래스 {len(self.genre_classes)}개 로드 완료")
            else:
                print(f"❌ 장르 메타데이터 파일을 찾을 수 없습니다: {self.genre_json_path}")

            # 분위기 클래스 로드
            if self.mood_json_path.exists():
                with open(self.mood_json_path, 'r', encoding='utf-8') as f:
                    mood_metadata = json.load(f)
                    self.mood_classes = mood_metadata.get('classes', [])
                    print(f"✅ 분위기 클래스 {len(self.mood_classes)}개 로드 완료")
            else:
                print(f"❌ 분위기 메타데이터 파일을 찾을 수 없습니다: {self.mood_json_path}")

        except Exception as e:
            print(f"❌ 메타데이터 로드 실패: {e}")

    def _load_pb_model(self, model_path):
        """TensorFlow .pb 모델 로드"""
        try:
            # GraphDef 로드
            with tf.io.gfile.GFile(str(model_path), "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

            # 새 그래프 생성 및 import
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(graph_def, name="")

            # 세션 생성
            session = tf.compat.v1.Session(graph=graph)

            return graph, session

        except Exception as e:
            print(f"❌ 모델 로드 실패 ({model_path}): {e}")
            return None, None

    def _load_models(self):
        """모델들 로드"""
        try:
            # 장르 분류 모델 로드
            if self.genre_model_path.exists():
                print("🔄 장르 분류 모델 로드 중...")
                self.genre_graph, self.genre_session = self._load_pb_model(self.genre_model_path)
                if self.genre_session is not None:
                    print("✅ 장르 분류 모델 로드 완료")
            else:
                print(f"❌ 장르 모델 파일을 찾을 수 없습니다: {self.genre_model_path}")

            # 분위기 분류 모델 로드
            if self.mood_model_path.exists():
                print("🔄 분위기 분류 모델 로드 중...")
                self.mood_graph, self.mood_session = self._load_pb_model(self.mood_model_path)
                if self.mood_session is not None:
                    print("✅ 분위기 분류 모델 로드 완료")
            else:
                print(f"❌ 분위기 모델 파일을 찾을 수 없습니다: {self.mood_model_path}")

        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")

    def preprocess_audio_essentia(self, audio, sr=16000):
        """
        Essentia 방식의 오디오 전처리 (audio_preprocessing.py 기반)
        """
        try:
            # 1. 리샘플링
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # 2. Essentia 정확한 파라미터
            frame_size = 512
            hop_size = 256
            n_mels = 96
            patch_size = 128

            # 3. 멜 스펙트로그램 (파워 스케일, 선형)
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

            # 4. Essentia 방식 로그 압축 (핵심!)
            # shift=1, scale=10000, log10
            # 더 안전한 로그 변환
            mel_bands_shifted = (mel_spec + 1e-8) * 10000  # 1 대신 1e-8 사용
            mel_bands_log = np.log10(mel_bands_shifted)

            # 정규화 추가
            mel_bands_log = (mel_bands_log - mel_bands_log.mean()) / mel_bands_log.std()

            # 5. 패치 생성
            n_frames = mel_bands_log.shape[1]
            patches = []
            patch_hop_size = 64  # patch_size // 2

            for start in range(0, n_frames - patch_size + 1, patch_hop_size):
                end = start + patch_size
                patch = mel_bands_log[:, start:end]  # [96, 128]
                patches.append(patch)

            # 마지막 패치 추가
            if len(patches) == 0 or n_frames >= patch_size:
                if n_frames >= patch_size:
                    last_patch = mel_bands_log[:, -patch_size:]
                    if len(patches) == 0 or not np.array_equal(patches[-1], last_patch):
                        patches.append(last_patch)

            # 6. 64개 패치로 조정
            if len(patches) > 64:
                indices = np.linspace(0, len(patches) - 1, 64, dtype=int)
                patches = [patches[i] for i in indices]
            elif len(patches) < 64:
                while len(patches) < 64:
                    patches.append(patches[-1].copy())

            # 7. 배치로 결합: [64, 96, 128] -> [64, 128, 96]
            mel_batch = np.array(patches, dtype=np.float32)
            mel_batch = np.transpose(mel_batch, (0, 2, 1))  # [64, 128, 96]

            return mel_batch

        except Exception as e:
            print(f"❌ 오디오 전처리 실패: {e}")
            return None

    def classify_genre(self, mel_spectrogram):
        """장르 분류 및 임베딩 추출"""
        if self.genre_session is None or mel_spectrogram is None:
            return None, None

        try:
            # 입력/출력 텐서 가져오기
            input_tensor = self.genre_graph.get_tensor_by_name(self.genre_input_name)
            predictions_tensor = self.genre_graph.get_tensor_by_name(self.genre_output_predictions)
            embeddings_tensor = self.genre_graph.get_tensor_by_name(self.genre_output_embeddings)

            # 추론 실행
            predictions, embeddings = self.genre_session.run(
                [predictions_tensor, embeddings_tensor],
                feed_dict={input_tensor: mel_spectrogram}
            )

            # 배치 차원에서 평균내기 (장르 예측만)
            predictions_mean = np.mean(predictions, axis=0)  # [400]
            # embeddings는 평균내지 않고 원본 [64, 1280] 그대로 반환

            # 상위 10개 장르 추출
            top_indices = np.argsort(predictions_mean)[-10:][::-1]

            genre_results = []
            for idx in top_indices:
                genre_name = self.genre_classes[idx] if idx < len(self.genre_classes) else f"Unknown_{idx}"
                genre_results.append({
                    'genre': genre_name,
                    'score': float(predictions_mean[idx]),
                    'index': int(idx)
                })

            return genre_results, embeddings  # embeddings_mean → embeddings

        except Exception as e:
            print(f"❌ 장르 분류 실패: {e}")
            return None, None

    def classify_mood(self, embeddings):
        """분위기/테마 분류 - 참고 코드 방식 적용"""
        if self.mood_session is None or embeddings is None:
            return None

        try:
            # embeddings는 [64, 1280] 형태
            # 각 패치별로 개별 예측 후 평균내기 (참고 코드 방식)

            input_tensor = self.mood_graph.get_tensor_by_name(self.mood_input_name)
            output_tensor = self.mood_graph.get_tensor_by_name(self.mood_output_name)

            all_patch_predictions = []

            # 각 패치별로 예측
            for i in range(embeddings.shape[0]):  # 64개 패치
                patch_embedding = embeddings[i:i + 1]  # [1, 1280]

                patch_prediction = self.mood_session.run(
                    output_tensor,
                    feed_dict={input_tensor: patch_embedding}
                )
                all_patch_predictions.append(patch_prediction[0])  # [56]

            # 패치별 예측을 평균내기 (참고 코드 방식)
            activation_avs = []
            for i in range(len(all_patch_predictions[0])):  # 56개 태그
                vals = [all_patch_predictions[j][i] for j in range(len(all_patch_predictions))]
                activation_avs.append(sum(vals) / len(vals))

            predictions = np.array(activation_avs)

            # IQR 기반 임계값 (참고 코드 방식)
            q1 = np.quantile(predictions, 0.25)
            q3 = np.quantile(predictions, 0.75)
            outlier_threshold = q3 + (1.5 * (q3 - q1))

            # 모든 예측 결과를 딕셔너리로 만들기
            all_predictions = {}
            prominent_moods = []

            for i, mood_class in enumerate(self.mood_classes):
                score = float(predictions[i])
                all_predictions[mood_class] = score

                if score >= outlier_threshold and mood_class != 'melodic':
                    prominent_moods.append({
                        'mood': mood_class,
                        'score': score,
                        'index': i
                    })

            # 점수순으로 정렬
            prominent_moods.sort(key=lambda x: x['score'], reverse=True)

            # 상위 10개 분위기
            top_moods = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                'prominent_moods': prominent_moods,
                'top_moods': top_moods,
                'all_predictions': all_predictions,
                'threshold': outlier_threshold
            }

        except Exception as e:
            print(f"❌ 분위기 분류 실패: {e}")
            return None

    def classify_music(self, audio_file):
        """
        음악 파일의 장르와 분위기를 종합적으로 분류

        Args:
            audio_file: 음악 파일 경로

        Returns:
            dict: 분류 결과
        """
        print(f"🎵 음악 분석 시작: {audio_file}")
        print("=" * 50)

        try:
            # 1. 오디오 로드
            print("🔄 오디오 로드 중...")
            audio, sr = librosa.load(audio_file, sr=16000)
            print(f"✅ 오디오 로드 완료: {len(audio) / sr:.1f}초")

            # 2. 전처리
            print("🔄 오디오 전처리 중...")
            mel_spectrogram = self.preprocess_audio_essentia(audio, sr)
            if mel_spectrogram is None:
                return {"error": "오디오 전처리 실패"}
            print(f"✅ 전처리 완료: {mel_spectrogram.shape}")

            # 3. 장르 분류 및 임베딩 추출
            print("🔄 장르 분류 중...")
            genre_results, embeddings = self.classify_genre(mel_spectrogram)
            if genre_results is None:
                return {"error": "장르 분류 실패"}
            print(f"✅ 장르 분류 완료: 상위 {len(genre_results)}개")

            # 4. 분위기 분류
            print("🔄 분위기 분류 중...")
            mood_results = self.classify_mood(embeddings)
            if mood_results is None:
                return {"error": "분위기 분류 실패"}
            print(f"✅ 분위기 분류 완료: {len(mood_results['prominent_moods'])}개 도출")

            # 5. 결과 정리
            result = {
                "file_path": audio_file,
                "audio_duration": len(audio) / sr,
                "preprocessing_shape": mel_spectrogram.shape,
                "genres": {
                    "top_genres": genre_results[:5],  # 상위 5개
                    "all_genres": genre_results
                },
                "moods": {
                    "prominent_moods": mood_results['prominent_moods'],
                    "top_moods": mood_results['top_moods'][:10],  # 상위 10개
                    "threshold": mood_results['threshold']
                },
                "model_info": {
                    "genre_model": "discogs-effnet-bs64-1",
                    "mood_model": "mtg_jamendo_moodtheme-discogs-effnet-1",
                    "total_genre_classes": len(self.genre_classes),
                    "total_mood_classes": len(self.mood_classes)
                }
            }

            return result

        except Exception as e:
            return {"error": f"분석 실패: {e}"}

    def close_sessions(self):
        """TensorFlow 세션 정리"""
        if self.genre_session:
            self.genre_session.close()
        if self.mood_session:
            self.mood_session.close()
        print("✅ TensorFlow 세션 정리 완료")


def main():
    """메인 실행 함수"""
    # 분류기 초기화
    classifier = MusicClassifier(model_dir="../models")

    # 테스트할 음악 파일
    music_file = "../musics/Debussy_-_Arabesque_-_Aufklarung.mp3"  # 실제 파일 경로로 변경하세요

    try:
        # 음악 분류 실행
        result = classifier.classify_music(music_file)

        # 결과 출력
        if "error" in result:
            print(f"❌ 오류: {result['error']}")
        else:
            print("\n" + "=" * 60)
            print("🎵 음악 분류 결과")
            print("=" * 60)

            print(f"📁 파일: {result['file_path']}")
            print(f"⏱️  길이: {result['audio_duration']:.1f}초")
            print(f"🔧 전처리 형태: {result['preprocessing_shape']}")

            print(f"\n🎼 상위 장르:")
            for i, genre_info in enumerate(result['genres']['top_genres'], 1):
                print(f"  {i}. {genre_info['genre']}: {genre_info['score']:.4f}")

            print(f"\n🎭 주요 분위기/테마 (임계값: {result['moods']['threshold']:.4f}):")
            if result['moods']['prominent_moods']:
                for i, mood_info in enumerate(result['moods']['prominent_moods'], 1):
                    print(f"  {i}. {mood_info['mood']}: {mood_info['score']:.4f}")
            else:
                print("  (임계값을 넘는 분위기 없음)")

            print(f"\n📊 상위 분위기/테마 (전체):")
            for i, (mood, score) in enumerate(result['moods']['top_moods'], 1):
                print(f"  {i}. {mood}: {score:.4f}")

            print(f"\n🤖 모델 정보:")
            print(
                f"  • 장르 모델: {result['model_info']['genre_model']} ({result['model_info']['total_genre_classes']}개 클래스)")
            print(
                f"  • 분위기 모델: {result['model_info']['mood_model']} ({result['model_info']['total_mood_classes']}개 클래스)")

    finally:
        # 세션 정리
        classifier.close_sessions()


if __name__ == "__main__":
    print("🎵 === 음악 장르 및 분위기 분류기 ===")
    print("Discogs EfficientNet + MTG Jamendo 모델 사용")
    print()

    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")