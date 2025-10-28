import glob
from urllib.parse import urlparse

import emoji


def load_blacklist_domains() -> set[str]:
    blacklist_domains = []
    dirs = [
        "adult",
        "phishing",
        "dating",
        "gambling",
        "filehosting",
        "ddos",
        "agressif",
        "chat",
        "mixed_adult",
        "arjel",
        "custom_domain",
    ]
    files = [f"ut1_blacklist/{d}/domains" for d in dirs]
    for file in files:
        with open(file, encoding="utf-8") as fp:
            domains = fp.read().split("\n")
        blacklist_domains.extend([d.strip() for d in domains if not len(d) == 0])
    blacklist_domains = set(blacklist_domains)
    return blacklist_domains


# blacklist_domains = load_blacklist_domains()


def is_blacklist_domain(url: str) -> bool:
    if urlparse(url).hostname in blacklist_domains:
        return True
    return False


def test_is_blacklist_domain():
    assert is_blacklist_domain("https://0-1avsex.com/test.jpeg") == True
    assert is_blacklist_domain("https://pics.dmm.co.jp//test.jpg") == True


def contain_ngword_in_url(url: str) -> bool:
    ng_words_url_path = "ng_words_url/word.txt"
    with open(ng_words_url_path, encoding="utf-8") as fp:
        ng_words = fp.read().split("\n")
    for ngword in ng_words:
        if ngword in url:
            return True
    return False


def test_contain_ngword_in_url():
    assert contain_ngword_in_url("https://porn.com") == True
    assert contain_ngword_in_url("https://adult.com") == True


def is_nsfw_url(url: str) -> bool:
    """Check if a URL is NSFW based on blacklist and NG words."""
    return is_blacklist_domain(url) or contain_ngword_in_url(url)


def load_ng_words() -> list[str]:
    ng_words = []
    files = glob.glob("ng_words/*.txt")
    for file in files:
        with open(file, encoding="utf-8") as fp:
            words = fp.read().split("\n")
        ng_words.extend([w.strip() for w in words if not len(w) == 0])
    return ng_words


ng_words = load_ng_words()


def contain_ngword(text: str) -> str | None:
    for ngword in ng_words:
        if ngword in text:
            return ngword
    return None


def test_contain_ngword():
    assert contain_ngword("18禁映画") == "18禁"
    assert contain_ngword("アダルト") == "アダルト"


# more than n repeated characters
def contain_repeated_characters(text: str, n: int = 4) -> bool:
    """Check if the text contains more than n repeated characters. if n=3, "笑ったwwww" returns True."""
    return any(text[i] * n in text for i in range(len(text) - n))


def test_contain_repeated_characters():
    assert contain_repeated_characters("笑ったwwww", 3) == True
    assert contain_repeated_characters("笑ったwwww", 4) == False


# contains a lot of \n
def contain_many_newlines(text: str, n: int = 3) -> bool:
    """Check if the text contains more than n newlines."""
    return text.count("\n") > n


def test_contain_many_newlines():
    assert contain_many_newlines("こんにちは\n\n\n", 2) == True
    assert contain_many_newlines("こんにちは\n\n\n", 3) == False
    assert contain_many_newlines("こんにちは\n\n\n\n", 3) == True


# contain a lot of emojis
def contain_many_emojis(text: str, n: int = 3) -> bool:
    """Check if the text contains more than n emojis."""
    return len([char for char in text if char in emoji.EMOJI_DATA]) > n


def test_contain_many_emojis():
    assert contain_many_emojis("✨🙏✨", 2) == True
    assert contain_many_emojis("✨🙏✨", 3) == False
    assert contain_many_emojis("😂😂😂😂", 3) == True
