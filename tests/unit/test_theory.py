from pathlib import Path

from physics_ai.theory import classify_theory_family, load_theory, validate_theory


def test_validate_theory_ok() -> None:
    ok, message = validate_theory(Path("examples/theories/einstein_hilbert.yaml"))
    assert ok
    assert "validated" in message.lower()


def test_classify_families() -> None:
    gr = load_theory(Path("examples/theories/einstein_hilbert.yaml"))
    fr = load_theory(Path("examples/theories/f_r.yaml"))
    assert classify_theory_family(gr) == "GR"
    assert classify_theory_family(fr) == "f(R)"
