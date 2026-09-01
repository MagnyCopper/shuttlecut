from shuttlecut.detector import (
    FramePersons,
    PersonBox,
    detect_persons,
    load_persons_jsonl,
    resolve_device,
)
from shuttlecut.sampler import extract_frames


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"
    assert resolve_device("auto") in ("mps", "cpu")


def test_detect_and_roundtrip(synth_video, tmp_path):
    frames = extract_frames(synth_video, str(tmp_path / "f"), fps=1.0, width=1280)[:3]
    jl = str(tmp_path / "persons.jsonl")
    rows = detect_persons(frames, frame_fps=1.0, device="cpu", out_jsonl=jl)
    assert all(isinstance(r, FramePersons) for r in rows)
    meta, loaded = load_persons_jsonl(jl)
    assert meta["frame_w"] == 1280 and len(loaded) == len(rows)
    assert all(p.conf >= 0.0 for r in loaded for p in r.persons)
