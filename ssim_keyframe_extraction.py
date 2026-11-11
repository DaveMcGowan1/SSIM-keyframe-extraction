"""Extract video frames for Stable Diffusion / EbSynth using SSIM."""

import os

import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Edit these settings before running the script.
INPUT_VIDEO = 'path/to/input/video.mp4'
FRAME_RATE = 1  # Maximum number of frames per second to compare (1 = once per second)
THRESHOLD = 0.90  # Keep a frame when SSIM against the last keyframe falls below this
OUTPUT_PATH = 'path/to/output/folder'

# Compare smaller copies for speed; save selected frames at their original resolution.
COMPARISON_WIDTH = 320


def comparison_frame(frame):
    """Create a small colour image for SSIM without modifying the export frame."""
    height, width = frame.shape[:2]
    if width > COMPARISON_WIDTH:
        new_height = max(1, round(height * COMPARISON_WIDTH / width))
        frame = cv2.resize(frame, (COMPARISON_WIDTH, new_height), interpolation=cv2.INTER_AREA)
    if min(frame.shape[:2]) < 7:
        raise ValueError('Video dimensions are too small for SSIM (minimum 7 pixels).')
    return frame


def extract_keyframes(input_video, output_path, frame_rate=FRAME_RATE, threshold=THRESHOLD):
    """Save the first frame and frames sufficiently different from the last saved one."""
    if frame_rate <= 0:
        raise ValueError('FRAME_RATE must be greater than zero.')
    if not -1 <= threshold <= 1:
        raise ValueError('THRESHOLD must be between -1 and 1 (SSIM similarity).')

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {input_video}')

    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not 0 < source_fps < float('inf'):
            raise ValueError('Could not read the source video frame rate.')

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        digits = len(str(max(0, total_frames - 1))) if total_frames > 0 else 6
        interval = max(1.0, source_fps / frame_rate)
        next_sample = 0.0
        saved_count = 0
        previous_keyframe = None

        synth_frames_path = os.path.join(output_path, 'synth_frames')
        os.makedirs(synth_frames_path, exist_ok=True)

        with tqdm(total=total_frames if total_frames > 0 else None,
                  desc='Processing frames') as progress:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                progress.update(1)

                if frame_index + 1e-9 >= next_sample:
                    next_sample += interval
                    candidate = comparison_frame(frame)
                    if (previous_keyframe is None or
                            ssim(previous_keyframe, candidate,
                                 channel_axis=2, data_range=255) < threshold):
                        filename = f'{frame_index:0{digits}d}.png'
                        filepath = os.path.join(synth_frames_path, filename)
                        if not cv2.imwrite(filepath, frame):
                            raise OSError(f'Could not save frame: {filepath}')
                        previous_keyframe = candidate
                        saved_count += 1
                frame_index += 1

        print(f'Saved {saved_count} keyframes to {synth_frames_path}')
        return saved_count
    finally:
        cap.release()


if __name__ == '__main__':
    extract_keyframes(INPUT_VIDEO, OUTPUT_PATH)
