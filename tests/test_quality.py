from oceanscribe_cleanup.quality import check_output, contains_phrase


def test_constraints_catch_semantic_changes_without_requiring_exact_punctuation():
    checks = {"required_groups": [["quarterly inspection"]], "forbidden": ["annual"]}
    assert check_output(
        "Quarterly inspection, yesterday.", "Quarterly inspection.", checks
    )["passed"]
    assert not check_output("Annual inspection.", "Quarterly inspection.", checks)["passed"]
    assert not check_output("Inspection.", "Quarterly inspection.", checks)["passed"]


def test_terminology_recall_false_insertions_and_preserve():
    checks = {"corrected_terms": ["JSON"], "absent_terms": ["OceanScribe"]}
    assert check_output("The JSON file is valid.", "", checks)["passed"]
    assert not check_output("The json file is valid.", "", checks)["passed"]
    assert not check_output("OceanScribe: the JSON file is valid.", "", checks)["passed"]
    assert not contains_phrase("annuality", "annual")
    assert not check_output("Done!", "Done.", {"preserve_exact": True})["passed"]
