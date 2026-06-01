from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_script() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "stage_epcam_af2m.py"
    spec = importlib.util.spec_from_file_location("stage_epcam_af2m", script_path)
    assert spec is not None
    assert spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules["stage_epcam_af2m"] = module
    spec.loader.exec_module(module)
    return module


def _write_epcam_fixture(data_dir: Path) -> None:
    data_dir.mkdir()
    (data_dir / "epcam_positives_filtered.csv").write_text(
        "vhh_no,vhh_sequence,cdr3_sequence,cluster_id\n"
        "1,QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAINSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAKGGYWGQGTQVTVSS,AA,1\n"
    )
    (data_dir / "epcam_scrambled_negatives.csv").write_text(
        "vhh_id,vhh_sequence,source_vhh_no,scramble_variant_idx,negative_type\n"
        "epcam-scr-1,QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAINSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAKGGYWGQGTQVTVSS,1,1,cdr_scramble\n"
    )
    (data_dir / "epcam_literature_negatives.csv").write_text(
        "vhh_id,sequence,target,source\n"
        "lit-1,QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAINSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAKGGYWGQGTQVTVSS,OtherTarget,paper\n"
    )


def test_write_metadata_keeps_epcam_labels_and_negative_context(tmp_path: Path) -> None:
    module = _load_script()
    data_dir = tmp_path / "epcam"
    _write_epcam_fixture(data_dir)
    examples = list(module.get_loader("epcam", data_dir=data_dir).load())

    metadata_path = tmp_path / "metadata.csv"
    module._write_metadata(metadata_path, examples)

    with metadata_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert module._label_counts(examples) == {"POS": 1, "SCR": 1, "OFF": 1}
    assert [row["label"] for row in rows] == ["POS", "SCR", "OFF"]
    assert rows[1]["source_vhh_no"] == "1"
    assert rows[2]["real_target"] == "OtherTarget"


def test_main_stages_manifest_without_submitting(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    module = _load_script()
    data_dir = tmp_path / "epcam"
    _write_epcam_fixture(data_dir)
    output_root = tmp_path / "predictions"
    staged_root = tmp_path / "staged"
    monkeypatch.setenv("MIRAGE_AF2M_OUTPUT_ROOT", str(output_root))
    monkeypatch.setenv("MIRAGE_AF2M_STAGED_ROOT", str(staged_root))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_epcam_af2m.py",
            "--data-dir",
            str(data_dir),
            "--chunk-size",
            "2",
            "--max-concurrent",
            "3",
        ],
    )

    assert module.main() == 0

    out = capsys.readouterr().out
    manifests = list(staged_root.glob("manifest_*_epcam.tsv"))
    metadata = list(staged_root.glob("manifest_*_epcam_metadata.csv"))
    assert len(manifests) == 1
    assert len(metadata) == 1
    assert "--array=0-1%3" in out
    assert "To submit after approval:" in out
    assert (staged_root / "fasta" / "epcam-pos-1.fasta").is_file()
