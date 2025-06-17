"""
Essentia TensorflowInputMusiCNN 완전 재현
Essentia Labs에서 확인된 정확한 로그 압축 방식 적용
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_essentia_exact(audio, sr=16000):
    """
    Essentia TensorflowInputMusiCNN 완전 재현
    shift=1, scale=10000, log10 적용
    """
    # 1. 리샘플링
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    print(f"🔍 오디오 길이: {len(audio)/sr:.2f}초")

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

    print(f"🔍 원본 멜 스펙트로그램:")
    print(f"   형태: {mel_spec.shape}")
    print(f"   범위: [{mel_spec.min():.6f}, {mel_spec.max():.6f}]")
    print(f"   평균: {mel_spec.mean():.6f}")

    # 4. Essentia 방식 로그 압축 (핵심!)
    # shift=1, scale=10000, log10
    mel_bands_shifted = (mel_spec + 1) * 10000
    mel_bands_log = np.log10(mel_bands_shifted)

    print(f"🔍 Essentia 로그 압축 후:")
    print(f"   형태: {mel_bands_log.shape}")
    print(f"   범위: [{mel_bands_log.min():.6f}, {mel_bands_log.max():.6f}]")
    print(f"   평균: {mel_bands_log.mean():.6f}")

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

    print(f"🔍 생성된 패치 수: {len(patches)}")

    # 6. 64개 패치로 조정
    if len(patches) > 64:
        indices = np.linspace(0, len(patches) - 1, 64, dtype=int)
        patches = [patches[i] for i in indices]
        print(f"🔍 패치 다운샘플링: {len(patches)}개로 축소")
    elif len(patches) < 64:
        original_count = len(patches)
        while len(patches) < 64:
            patches.append(patches[-1].copy())
        print(f"🔍 패치 패딩: {original_count}개 -> {len(patches)}개")

    # 7. 배치로 결합: [64, 96, 128] -> [64, 128, 96]
    mel_batch = np.array(patches, dtype=np.float32)
    mel_batch = np.transpose(mel_batch, (0, 2, 1))  # [64, 128, 96]

    print(f"🔍 최종 배치:")
    print(f"   형태: {mel_batch.shape}")
    print(f"   범위: [{mel_batch.min():.6f}, {mel_batch.max():.6f}]")
    print(f"   평균: {mel_batch.mean():.6f}")
    print(f"   표준편차: {mel_batch.std():.6f}")

    return mel_batch


def test_correct_preprocessing(audio_file):
    """
    수정된 전처리 테스트
    """
    print("🎵 === Essentia 완전 재현 전처리 테스트 ===")
    print(f"📁 파일: {audio_file}")
    print("=" * 60)

    try:
        # 오디오 로드
        audio, sr = librosa.load(audio_file, sr=16000)

        # 완전히 수정된 전처리
        mel_batch = preprocess_essentia_exact(audio, sr)

        # 결과 분석
        print(f"\n📊 전처리 결과 분석:")
        print(f"   최종 형태: {mel_batch.shape}")
        print(f"   값 범위: [{mel_batch.min():.6f}, {mel_batch.max():.6f}]")
        print(f"   평균: {mel_batch.mean():.6f}")
        print(f"   표준편차: {mel_batch.std():.6f}")

        # 예상 범위 확인
        if 0 <= mel_batch.min() and mel_batch.max() <= 6:
            print("✅ 값 범위가 Essentia 예상 범위 [0, 6]에 맞습니다!")
        else:
            print("❌ 값 범위가 Essentia 예상 범위를 벗어납니다.")

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Essentia 완전 재현 전처리 결과', fontsize=16)

        for i in range(min(4, mel_batch.shape[0])):
            row, col = i // 2, i % 2
            patch = mel_batch[i].T  # [96, 128] -> [128, 96]로 전치

            im = axes[row, col].imshow(
                patch,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                vmin=0,
                vmax=6,  # Essentia 범위
                interpolation='nearest'
            )
            axes[row, col].set_title(f'패치 {i+1}')
            axes[row, col].set_xlabel('멜 주파수 빈')
            axes[row, col].set_ylabel('시간 프레임')
            plt.colorbar(im, ax=axes[row, col])

        plt.tight_layout()
        plt.savefig('essentia_exact_preprocessing.png', dpi=150, bbox_inches='tight')
        print("✅ 시각화 저장됨: essentia_exact_preprocessing.png")

        return mel_batch

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return None


def compare_all_methods(audio_file):
    """
    모든 전처리 방법 비교
    """
    print("🔄 모든 전처리 방법 비교")
    print("=" * 60)

    try:
        audio, sr = librosa.load(audio_file, sr=16000)

        # 방법 1: 기존 dB 방식
        mel_spec_db = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=96, n_fft=512, hop_length=256,
            fmin=0.0, fmax=8000.0, power=2.0, norm='slaney', htk=False
        )
        mel_spec_db = librosa.power_to_db(mel_spec_db, ref=np.max)

        # 방법 2: Essentia 정확한 방식
        mel_spec_essentia = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=96, n_fft=512, hop_length=256,
            fmin=0.0, fmax=8000.0, power=2.0, norm='slaney', htk=False
        )
        mel_bands_shifted = (mel_spec_essentia + 1) * 10000
        mel_bands_log = np.log10(mel_bands_shifted)

        print(f"📊 비교 결과:")
        print(f"   기존 dB 방식:")
        print(f"      범위: [{mel_spec_db.min():.3f}, {mel_spec_db.max():.3f}]")
        print(f"      평균: {mel_spec_db.mean():.3f}")

        print(f"   Essentia 정확한 방식:")
        print(f"      범위: [{mel_bands_log.min():.3f}, {mel_bands_log.max():.3f}]")
        print(f"      평균: {mel_bands_log.mean():.3f}")

        # 범위 확인
        if 0 <= mel_bands_log.min() and mel_bands_log.max() <= 6:
            print("✅ Essentia 방식이 예상 범위 [0, 6]에 맞습니다!")
        else:
            print(f"❌ Essentia 방식도 범위가 맞지 않습니다: [{mel_bands_log.min():.3f}, {mel_bands_log.max():.3f}]")

    except Exception as e:
        print(f"❌ 비교 실패: {e}")


if __name__ == "__main__":
    # 테스트 파일
    test_file = "../musics/Anitek_-_Tab_+_Anitek_-_Bleach.mp3"

    if Path(test_file).exists():
        # 완전히 수정된 전처리 테스트
        test_correct_preprocessing(test_file)

        # 모든 방법 비교
        compare_all_methods(test_file)
    else:
        print(f"❌ 테스트 파일을 찾을 수 없습니다: {test_file}")