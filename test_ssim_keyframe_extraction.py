"""Small deterministic regression tests for the SSIM keyframe picker."""

import numpy as np
import pytest

import ssim_keyframe_extraction as extractor


class FakeCapture:
    def __init__(self, frames, fps=4):
        self.frames = frames
        self.fps = fps
        self.position = 0
        self.released = False

    def isOpened(self):
        return bool(self.frames)

    def get(self, prop):
        if prop == extractor.cv2.CAP_PROP_FPS:
            return self.fps
        if prop == extractor.cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


def run_fake(monkeypatch, tmp_path, frames, fps=4, sample_rate=4, threshold=0.995):
    video = FakeCapture(frames, fps=fps)
    written = []

    monkeypatch.setattr(extractor.cv2, 'VideoCapture', lambda path: video)

    def write_file(path, frame):
        written.append((path, frame.copy()))
        return True

    monkeypatch.setattr(extractor.cv2, 'imwrite', write_file)
    count = extractor.extract_keyframes(
        'placeholder.mp4', str(tmp_path), frame_rate=sample_rate, threshold=threshold
    )
    return count, written, video


def frame(value):
    return np.full((32, 32, 3), value, dtype=np.uint8)


def test_static_video_only_exports_first_frame(monkeypatch, tmp_path):
    count, written, video = run_fake(
        monkeypatch, tmp_path, [frame(100)] * 8
    )
    assert count == 1
    assert written[0][0].endswith('0.png')
    assert video.released


def test_gradual_change_accumulates_from_last_keyframe(monkeypatch, tmp_path):
    # Adjacent frames are close, but frame 5 is sufficiently different
    # from frame 0 to require a new keyframe.
    frames = [frame(value) for value in [100, 102, 104, 106, 108, 110]]
    count, written, _ = run_fake(
        monkeypatch, tmp_path, frames, threshold=0.997
    )
    assert count >= 2
    assert written[0][0].endswith('0.png')
    assert any(not path.endswith('0.png') for path, _ in written)


def test_frame_rate_is_sampling_rate_not_delay(monkeypatch, tmp_path):
    # 12 frames at 4 fps, sampling at 1 fps: indices 0, 4, 8.
    frames = [frame(0 if i % 4 == 0 else 255) for i in range(12)]
    count, written, _ = run_fake(
        monkeypatch, tmp_path, frames, fps=4, sample_rate=1, threshold=0.9
    )
    assert count == 1
    assert [p.rsplit('/', 1)[-1] for p, _ in written] == ['00.png']


def test_missing_video_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(extractor.cv2, 'VideoCapture', lambda path: FakeCapture([]))
    with pytest.raises(ValueError, match='Could not open'):
        extractor.extract_keyframes('missing.mp4', str(tmp_path))


def test_invalid_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='FRAME_RATE'):
        extractor.extract_keyframes('x.mp4', str(tmp_path), frame_rate=0)
    with pytest.raises(ValueError, match='THRESHOLD'):
        extractor.extract_keyframes('x.mp4', str(tmp_path), threshold=2)
