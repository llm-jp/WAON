from fast_langdetect import detect
from lingua import LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_spoken_languages().build()


def is_japanese(text: str) -> bool:
    language = detector.detect_language_of(text)
    if language is None:
        return False
    return language.iso_code_639_1.name == "JA"


def is_japanese_fast(text: str) -> bool:
    text = text.replace("\n", " ")
    if detect(text)["lang"] == "ja":
        return True
    return False


def test_is_japanese():
    assert is_japanese("こんにちは") == True
    assert is_japanese("hello") == False
    assert is_japanese("你好") == False
    assert is_japanese("안녕하세요") == False


def test_is_japanese_fast():
    assert is_japanese_fast("こんにちは") == True
    assert is_japanese_fast("hello") == False
    assert is_japanese_fast("你好") == False
    assert is_japanese_fast("안녕하세요") == False
    assert is_japanese_fast("こんにちは, は英語で言うとhelloです") == True
