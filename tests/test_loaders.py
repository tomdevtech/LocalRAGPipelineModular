import pytest

from loaders import load_reviews_csv


def test_load_reviews_csv(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text(
        "Title,Date,Rating,Review\n"
        'Great,2024-01-01,5,"Loved it, truly"\n'
        ",,,\n"
        "NoRating,2024-02-02,,Fine\n"
    )
    docs = load_reviews_csv(str(csv))

    assert len(docs) == 2  # the empty row is skipped
    assert docs[0].page_content == "Great Loved it, truly"
    assert docs[0].metadata == {"source": "r.csv", "row": 0, "date": "2024-01-01", "rating": 5}
    assert type(docs[0].metadata["rating"]) is int  # not numpy.int64
    assert "rating" not in docs[1].metadata  # NaN is not a valid Chroma value


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reviews_csv(str(tmp_path / "nope.csv"))


def test_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("Title,Text\na,b\n")
    with pytest.raises(ValueError, match="Date, Rating, Review"):
        load_reviews_csv(str(csv))
