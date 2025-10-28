from huggingface_hub import hf_hub_download
import fasttext
import MeCab


class QualityClassifier:
    def __init__(self):
        self.model = fasttext.load_model(
            hf_hub_download("llm-jp/waon-quality-classifier", "quality_classifier.bin")
        )
        self.mecab = MeCab.Tagger("-Owakati")

    def classify(self, text: str) -> float:
        return self.predict_quality(text)

    def preprocess_ja(self, text: str) -> str:
        """改行削除 + MeCabによる分かち書き"""
        text = text.replace("\n", " ").replace("\r", " ").strip()
        return self.mecab.parse(text).strip()

    def predict_quality(self, text: str) -> float:
        preprocessed = self.preprocess_ja(text)
        logits = self.model.predict(preprocessed, k=-1)
        score = sum(
            [int(label[-1]) * prob for label, prob in zip(logits[0], logits[1])]
        )
        return score


class EduWikiClassifier:
    def __init__(self):
        self.model = fasttext.load_model(
            hf_hub_download("tokyotech-llm/edu-classifier", "wiki.bin")
        )

    def classify(self, text: str) -> float:
        return self.wiki_based_classifier(text)

    def wiki_based_classifier(self, text: str) -> float:
        text = text.replace("\n", " ")
        res = self.model.predict(text, k=-1)
        edu_score = res[1][0] if res[0][0] == "__label__pos" else 1 - res[1][0]
        return edu_score


class EduLLMClassifier:
    def __init__(self):
        self.model = fasttext.load_model(
            hf_hub_download("tokyotech-llm/edu-classifier", "llm_llama.bin")
        )

    def classify(self, text: str) -> float:
        return self.llm_based_classifier(text)

    def llm_based_classifier(self, text: str) -> float:
        text = text.replace("\n", " ")
        res = self.model.predict(text, k=-1)
        edu_score = sum([int(label[-1]) * prob for label, prob in zip(res[0], res[1])])
        return edu_score


if __name__ == "__main__":
    wiki_classifier = EduWikiClassifier()
    llm_classifier = EduLLMClassifier()
    quality_classifier = QualityClassifier()
    good_text = "京都大学は、日本の京都府に位置する国立大学で、1877年に設立されました。"
    bad_text = "京都大学。京都府。日本の国立大学。1877年設立。"
    print(
        f"text: {good_text}, wiki_score: {wiki_classifier.classify(good_text)}, llm_score: {llm_classifier.classify(good_text)}"
    )
    print(
        f"text: {bad_text}, wiki_score: {wiki_classifier.classify(bad_text)}, llm_score: {llm_classifier.classify(bad_text)}"
    )
    print(f"text: {good_text}, quality_score: {quality_classifier.classify(good_text)}")
    print(f"text: {bad_text}, quality_score: {quality_classifier.classify(bad_text)}")
