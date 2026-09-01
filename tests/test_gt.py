import pytest

from shuttlecut.labeling.gt import GTError, load_gt, save_gt


def test_gt_roundtrip(tmp_path):
    p = str(tmp_path / "g.json")
    save_gt(p, "video1", [{"id": 1, "start_s": 5.0, "end_s": 12.0, "note": ""}])
    d = load_gt(p)
    assert d["video"] == "video1" and d["rallies"][0]["start_s"] == 5.0


def test_gt_rejects_overlap(tmp_path):
    p = str(tmp_path / "g.json")
    pth = tmp_path / "g.json"
    pth.write_text('{"rallies": [{"start_s": 5.0, "end_s": 12.0}, {"start_s": 10.0, "end_s": 15.0}]}')
    with pytest.raises(GTError):
        load_gt(p)


def test_gt_rejects_bad_order(tmp_path):
    p = str(tmp_path / "g.json")
    tmp_path.joinpath("g.json").write_text('{"rallies": [{"start_s": 12.0, "end_s": 10.0}]}')
    with pytest.raises(GTError):
        load_gt(p)


def test_save_gt_rejects_empty_without_writing(tmp_path):
    p = tmp_path / "g.json"
    with pytest.raises(GTError):
        save_gt(str(p), "video1", [])
    assert not p.exists()


def test_save_gt_rejects_unsorted_without_writing(tmp_path):
    p = tmp_path / "g.json"
    rallies = [{"start_s": 5.0, "end_s": 8.0}, {"start_s": 1.0, "end_s": 3.0}]
    with pytest.raises(GTError):
        save_gt(str(p), "video1", rallies)
    assert not p.exists()


def test_save_gt_rejects_overlap_without_writing(tmp_path):
    p = tmp_path / "g.json"
    rallies = [{"start_s": 1.0, "end_s": 5.0}, {"start_s": 4.0, "end_s": 8.0}]
    with pytest.raises(GTError):
        save_gt(str(p), "video1", rallies)
    assert not p.exists()
