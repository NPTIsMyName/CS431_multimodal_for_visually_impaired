import time
import os
import torch
import librosa
from transformers import pipeline
from faster_whisper import WhisperModel

# =========================
# <--- CẤU HÌNH --->
# =========================
AUDIO_PATH = "test_data/chi_cho_tao_cai_chai_mau_do.wav"
# AUDIO_PATH = "test_data/cai_nay_la_cai_gi_vay.wav"

# Đổi giá trị này thành "tiny", "small", hoặc "base" để test các bản khác nhau
TARGET_MODEL_SIZE = "base" 

# =========================
# BASE MODEL (HF - original)
# =========================
def run_base_model(audio_path, size):
    hf_model_id = f"vinai/PhoWhisper-{size}"
    print(f"\n[BASE MODEL] {hf_model_id} (Transformers)")

    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "automatic-speech-recognition",
        model=hf_model_id,
        device=device
    )

    # Đọc file audio bằng librosa và ép tần số lấy mẫu về 16000Hz (chuẩn của Whisper)
    print("[-] Đang nạp audio vào RAM...")
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

    start = time.time()
    # Truyền dữ liệu thô (raw data) vào pipeline thay vì đường dẫn file
    result = pipe({"raw": audio_array, "sampling_rate": sampling_rate})
    latency = time.time() - start

    text = result["text"]

    return text, latency


# =========================
# CT2 MODEL (quantized)
# =========================
def run_ct2_model(audio_path, size):
    ct2_model_path = f"phowhisper-{size}-ct2"
    print(f"\n[CT2 MODEL] {ct2_model_path} (CTranslate2 int8)")
    
    # Kiểm tra xem bạn đã convert model CT2 cho size này chưa
    if not os.path.exists(ct2_model_path):
        print(f"[!] LỖI: Không tìm thấy thư mục '{ct2_model_path}'.")
        print(f"[*] Vui lòng chạy lệnh convert trước:\n    ct2-transformers-converter --model vinai/PhoWhisper-{size} --output_dir {ct2_model_path} --copy_files tokenizer.json preprocessor_config.json --quantization int8")
        return "[Lỗi thiếu Model CT2]", 0.0

    model = WhisperModel(
        ct2_model_path,
        device="cpu",          # CPU benchmark cho công bằng
        compute_type="int8"    # quantized
    )

    start = time.time()

    segments, _ = model.transcribe(
        audio_path,
        language="vi",
        beam_size=1,
        condition_on_previous_text=False
    )

    text = "".join([s.text for s in segments])
    latency = time.time() - start

    return text, latency


# =========================
# COMPARE FUNCTION
# =========================
def compare(size):
    print("\n" + "=" * 70)
    print(f"STT BENCHMARK: KÍCH THƯỚC [{size.upper()}] - BASE vs CT2 QUANTIZED")
    print("=" * 70)

    base_text, base_time = run_base_model(AUDIO_PATH, size)
    ct2_text, ct2_time = run_ct2_model(AUDIO_PATH, size)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"[BASE MODEL] Latency: {base_time:.4f}s")
    print(f"[CT2 INT8]   Latency: {ct2_time:.4f}s")

    print("\n--- BASE OUTPUT ---")
    print(base_text)

    print("\n--- CT2 OUTPUT ---")
    print(ct2_text)

    if ct2_time > 0 and base_time > 0:
        speedup = base_time / ct2_time
        print("\n[PERFORMANCE]")
        print(f"Speedup CT2: ~{speedup:.2f}x faster")

    print("=" * 70)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if os.path.exists(AUDIO_PATH):
        compare(TARGET_MODEL_SIZE)
    else:
        print(f"[!] Không tìm thấy file audio: {AUDIO_PATH}")