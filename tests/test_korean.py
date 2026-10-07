from malmoi import korean


def test_romanization_follows_pronunciation():
    cases = {
        "국물": "gungmul",     # nasalization
        "신라": "silla",       # liquidization
        "같이": "gachi",       # palatalization
        "설날": "seollal",
        "심리": "simni",
        "좋아": "joa",         # ㅎ drops before a vowel
        "킹받네": "kingbanne",
        "안녕하세요": "annyeonghaseyo",
    }
    for word, expected in cases.items():
        assert korean.romanize(word) == expected, word


def test_romanize_keeps_non_hangul():
    assert korean.romanize("눈치 (nunchi)!") == "nunchi (nunchi)!"


def test_headword_prefers_dictionary_pronunciation():
    assert korean.romanize_headword("국물", "궁ː물") == "gungmul"
    assert korean.romanize_headword("눈치", "") == "nunchi"


def test_lemmas_handle_conjugation():
    assert "걱정하다" in korean.content_lemmas("너무 걱정했어")
    assert "망설이다" in korean.content_lemmas("그냥 망설여져서 못 했어요")


def test_target_used():
    assert korean.target_used("걱정하다", "어제 좀 걱정했어")["used"]
    assert korean.target_used("눈치 보다", "눈치 보지 마")["used"]
    assert not korean.target_used("서운하다", "좀 섭섭했어")["used"]


def test_speech_level():
    assert korean.detect_speech_level("고마워!") == "casual"
    assert korean.detect_speech_level("내일 봬요~") == "polite"
    assert korean.detect_speech_level("네 알겠습니다 팀장님") == "formal"
    assert korean.level_ok("polite", "formal")
    assert not korean.level_ok("polite", "casual")
