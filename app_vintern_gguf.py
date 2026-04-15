import gradio as gr
import time
import os
import torch
import numpy as np
import soundfile as sf
from PIL import Image

# TỐI ƯU HÓA TÀI NGUYÊN CPU
threads = os.cpu_count()
torch.set_num_threads(threads)
print(f"[*] Đã ép PyTorch sử dụng {threads} luồng CPU.")

# Import các module đã tối ưu
from fastwhis import FastSpeechToText
from tts_vieneu import FastTextToSpeech
from vintern_pipeline_gguf import VisionLanguageModel

print("=" * 60)
print("ĐANG TẢI CÁC MÔ HÌNH VÀO BỘ NHỚ (RAM)...")
print("=" * 60)

stt_module = FastSpeechToText(model_size="base")
vlm_module = VisionLanguageModel()
tts_module = FastTextToSpeech(voice_index=0)


def warm_up_models():
    """
    Khởi động nóng để tránh delay lần đầu
    """
    print("\n[*] Đang chạy Warm-up...")

    dummy_audio_path = "dummy_warmup.wav"
    sf.write(dummy_audio_path, np.zeros(16000), 16000)

    dummy_image_path = "dummy_warmup.jpg"
    Image.new("RGB", (448, 448)).save(dummy_image_path)

    try:
        stt_module.transcribe(dummy_audio_path)
        vlm_module.analyze(dummy_image_path, "Test")
        print("[+] Warm-up hoàn tất!")
    except Exception as e:
        print(f"[!] Lỗi Warm-up: {e}")
    finally:
        if os.path.exists(dummy_audio_path):
            os.remove(dummy_audio_path)
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)


warm_up_models()


def process_multimodal(image_path, audio_path):
    if not image_path or not audio_path:
        return "⚠️ Vui lòng cung cấp đầy đủ cả hình ảnh và âm thanh!", None

    start_time = time.time()

    # 1. STT
    stt_start_time = time.time()
    question = stt_module.transcribe(audio_path)
    stt_latency = time.time() - stt_start_time

    if not question:
        return "❌ Không nghe rõ câu hỏi. Vui lòng thu âm lại.", None

    # 2. VLM
    vlm_start_time = time.time()
    answer = vlm_module.analyze(image_path, question)
    vlm_latency = time.time() - vlm_start_time

    # 3. TTS
    output_audio_path = "output_response.wav"
    tts_latency = 0

    try:
        print("\n[*] Đang sinh âm thanh phản hồi...")
        tts_start_time = time.time()

        audio_spec = tts_module.tts.infer(text=answer)
        tts_module.tts.save(audio_spec, output_audio_path)

        tts_latency = time.time() - tts_start_time
        print(f"[TTS Latency] {tts_latency:.4f}s")

    except Exception as e:
        print(f"[!] Lỗi TTS: {e}")
        output_audio_path = None

    total_latency = time.time() - start_time

    final_text = (
        f"🗣️ **Người dùng hỏi:** {question}\n\n"
        f"🤖 **Trợ lý trả lời:** {answer}\n\n"
        f"⏱️ *Tổng thời gian:* {total_latency:.2f}s\n"
        f"   *(ASR: {stt_latency:.2f}s | VLM: {vlm_latency:.2f}s | TTS: {tts_latency:.2f}s)*"
    )

    return final_text, output_audio_path


# UI Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🕶️ VibeBlind Assistant
        **Trợ lý AI đa phương thức hỗ trợ người khiếm thị (Offline CPU)**
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Input")
            img_in = gr.Image(type="filepath", label="Ảnh")
            audio_in = gr.Audio(type="filepath", sources=["microphone", "upload"], label="Audio")
            btn_submit = gr.Button("🔍 Phân tích", variant="primary")

        with gr.Column():
            gr.Markdown("### 2. Output")
            text_out = gr.Markdown()
            audio_out = gr.Audio(autoplay=True)

    btn_submit.click(
        fn=process_multimodal,
        inputs=[img_in, audio_in],
        outputs=[text_out, audio_out],
    )


if __name__ == "__main__":
    print("\n[+] Web đang chạy...")
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)