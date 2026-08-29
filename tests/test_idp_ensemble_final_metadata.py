from scripts.discover_idp_ensemble_final_metadata import record_text


def test_record_text_includes_title_and_entity_descriptions():
    text = record_text({
        "title": "anti-tau Fab",
        "polymer_entities": [{"description": "microtubule-associated protein tau"}],
    })
    assert "anti-tau Fab" in text
    assert "microtubule-associated protein tau" in text
