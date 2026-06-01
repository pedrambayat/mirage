import mirage


def test_version_string() -> None:
    assert isinstance(mirage.__version__, str)
    assert mirage.__version__.count(".") == 2
