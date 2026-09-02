from shuttlecut.eval.evaluate import evaluate


def test_perfect_match():
    gt = [(10.0, 20.0), (30.0, 40.0)]
    det = [(10.2, 19.8), (30.5, 40.4)]
    r = evaluate(det, gt)
    assert r.recall == 1.0 and r.precision == 1.0
    assert not r.missed and not r.extra and r.mae_s < 0.5


def test_missed_and_extra():
    gt = [(10.0, 20.0), (30.0, 40.0)]
    det = [(10.0, 20.0), (60.0, 70.0)]
    r = evaluate(det, gt)
    assert r.recall == 0.5 and r.precision == 0.5
    assert (30.0, 40.0) in r.missed
    assert (60.0, 70.0) in r.extra


def test_boundary_shift_still_matches():
    gt = [(10.0, 20.0)]
    det = [(12.2, 19.0)]
    r = evaluate(det, gt)
    assert r.recall == 1.0
    assert len(r.boundary) == 1


def test_fragment_classification_for_positive_overlap():
    gt = [(10.0, 30.0)]
    det = [(10.2, 17.8), (18.5, 26.0)]
    r = evaluate(det, gt, tol_s=15.0, overlap=0.3)
    assert len(r.fragment) == 1
    assert len(r.extra) == 0


def test_non_overlapping_unmatched_detection_is_extra():
    gt = [(10.0, 30.0)]
    det = [(10.2, 17.8), (18.5, 26.0), (60.0, 70.0)]
    r = evaluate(det, gt, tol_s=15.0, overlap=0.3)
    assert len(r.fragment) == 1
    assert len(r.extra) == 1
