import numpy as np
import os
import soundfile as sf
import onnxruntime as ort
from scipy.signal import resample
from scripts.inference_vr import infer
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

input_audio_path = "test"
model_path = "vr/hnsep_VR_44.1k_hop512_2024.05.onnx"
output_folder_path = "test_features"
chunk_size = 10
overlap_size = 8192
cut_size = 2048
batch_size = 8
save_png = True

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

def load_audio(file_path, target_sr=44100):
    audio, sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resample(audio, int(len(audio) * target_sr / sr))
    return target_sr, audio.astype("float32")

def save_audio(file_path, audio, sr):
    sf.write(file_path, audio, sr, subtype="PCM_16")

def interpolate_f0(times, f0_values):
    f0_values = np.array(f0_values)
    valid_indices = ~np.isnan(f0_values)
    if not np.any(valid_indices):
        return f0_values
    interpolated_f0 = np.interp(times, times[valid_indices], f0_values[valid_indices])
    return interpolated_f0

def extract_f0(audio_array, sr):
    sound = parselmouth.Sound(audio_array, sampling_frequency=sr)
    pitch = call(sound, "To Pitch", 0.0, 30, 1200)
    num_frames = call(pitch, "Get number of frames")
    
    f0_times = []
    f0_values = []

    for i in range(1, num_frames + 1):
        time = call(pitch, "Get time from frame number", i)
        f0 = call(pitch, "Get value in frame", i, "Hertz")
        if f0 is None:
            f0 = np.nan
        f0_times.append(time)
        f0_values.append(f0)

    f0_values = interpolate_f0(np.array(f0_times), f0_values)
    return f0_times, f0_values

def visualize_f0_overlay_combined(audio_array, sr, f0_times, f0_values, output_png_path):
    freqs, times, Sxx = spectrogram(audio_array, sr, nperseg=1024, noverlap=512, scaling="spectrum")
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, freqs, Sxx_dB, shading="gouraud", cmap="viridis")
    plt.colorbar(label="Amplitude (dB)")
    plt.plot(f0_times, f0_values, color="red", label="F0", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram with F0 Overlay")
    plt.legend(loc="upper right")
    plt.ylim(0, min(sr // 2, 3000))
    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()

def process_audio_file(file_path, session):
    file_basename = os.path.basename(file_path)
    fname, _ = os.path.splitext(file_basename)
    sr, waveform = load_audio(file_path)

    harmonic, noise = infer(
        session=session,
        waveform=waveform,
        chunk_size=chunk_size * 44100,
        overlap_size=overlap_size,
        cut_size=cut_size,
        batch_size=batch_size
    )

    harmonic = (harmonic * 32768).astype("int16")
    noise = (noise * 32768).astype("int16")

    harmonic_path = f"{output_folder_path}/{fname}_harmonic.wav"
    noise_path = f"{output_folder_path}/{fname}_noise.wav"
    save_audio(harmonic_path, harmonic, sr)
    save_audio(noise_path, noise, sr)

    f0_path = f"{output_folder_path}/{fname}_f0.txt"
    f0_times, f0_values = extract_f0(harmonic / 32768.0, sr)
    with open(f0_path, "w") as f:
        for time, f0 in zip(f0_times, f0_values):
            f.write(f"{time:.3f}\t{f0:.3f}\n")

    if save_png:
        f0_png_path = f"{output_folder_path}/{fname}_f0_overlay.png"
        combined_audio = (harmonic + noise).astype("float32") / 32768.0
        visualize_f0_overlay_combined(combined_audio, sr, f0_times, f0_values, f0_png_path)

def main():
    if os.path.isfile(input_audio_path):
        input_files = [input_audio_path]
    elif os.path.isdir(input_audio_path):
        input_files = [os.path.join(input_audio_path, f) for f in os.listdir(input_audio_path) if f.endswith(".wav")]
    else:
        raise ValueError("Input path must be a .wav file or a folder containing .wav files.")

    session = ort.InferenceSession(model_path)

    for file_path in input_files:
        print(f"Processing {file_path}...")
        process_audio_file(file_path, session)

    print("Processing complete.")

if __name__ == "__main__":
    main()