from app.agent.annotation.annotation_doc_reader import compute_file_key, normalize_relative_path


def test_file_key_matches_frontend_algorithm():
    assert normalize_relative_path("data\\2.jpg") == "data/2.jpg"
    key = compute_file_key("data/2.jpg")
    assert len(key) == 64
    assert key == compute_file_key("data\\2.jpg")
