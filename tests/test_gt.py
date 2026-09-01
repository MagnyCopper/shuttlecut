import pytest

from shuttlecut.labeling.gt import GTError, load_gt, save_gt


def test_gt_roundtrip(tmp_path):
    p = str(tmp_path / "g.json")
    save_gt(p, "video1", [{"id": 1, "start_s": 5.0, "end_s": 12.0, "note": ""}])
    d = load_gt(p)
    assert d["video"] == "video1" and d["rallies"][0]["start_s"] == 5.0


def test_gt_rejects_overlap(tmp_path):
    p = str(tmp_path / "g.json")
    save_gt(p, "video1", [{"id": 1, "start_s": 5.0, "end_s": 12.0, "note": ""},
                          {"id": 2, "start_s": 10.0, "end_s": 15.0, "note": ""}])
    with pytest.raises(GTError):
        load_gt(p)


def test_gt_rejects_bad_order(tmp_path):
    p = str(tmp_path / "g.json")
    save_gt(p, "video1", [{"id": 1, "start_s": 12.0, "end_s": 10.0, "note": ""}])
    with pytest.raises(GTError):
        load_gt(p)
