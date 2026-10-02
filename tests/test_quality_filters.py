import gzip
import json

from dataflow.data import Document
from dataflow.pipeline.filters import FineWebQualityFilter, GopherRepetitionFilter, telugu_quality_filters
from dataflow.pipeline.filters.gopher_repetition import duplicated_ngram_chars, find_duplicates, top_ngram_chars
from dataflow.utils.text import ends_sentence, split_indic_words

VILLAGE = (
    "రామాపురం ఆంధ్రప్రదేశ్ రాష్ట్రంలోని గుంటూరు జిల్లాకు చెందిన ఒక గ్రామం. "
    "ఈ గ్రామం మండల కేంద్రం నుండి పన్నెండు కిలోమీటర్ల దూరంలో ఉంది. "
    "గ్రామంలో ప్రాథమిక పాఠశాల, పంచాయతీ కార్యాలయం మరియు ఒక చిన్న గ్రంథాలయం ఉన్నాయి. "
    "ప్రజలు ఎక్కువగా వ్యవసాయం మీద ఆధారపడి జీవిస్తున్నారు. "
    "వరి, పత్తి, మిరప ఇక్కడ పండించే ప్రధాన పంటలు. "
    "ప్రతి సంవత్సరం సంక్రాంతి పండుగను గ్రామస్తులు ఘనంగా జరుపుకుంటారు. "
    "సమీప పట్టణానికి రోజూ బస్సు సౌకర్యం అందుబాటులో ఉంది. "
    "గ్రామానికి దగ్గరలో ఒక పెద్ద చెరువు ఉంది, దాని నీటితో పొలాలకు సాగునీరు అందుతుంది."
)
ENGLISH = (
    "Rivers carry water from distant mountains toward the sea, shaping valleys over thousands of years. "
    "Farmers settle along their banks because the soil there stays rich and easy to plough. "
    "Over time small villages become market towns, and bridges replace the old wooden ferries. "
    "Floods still arrive some summers, so engineers build levees and careful warning systems. "
    "Children learn to swim in shallow bends while fishermen mend their nets on the shore. "
    "Visitors often say the slow current makes these places feel calm and unhurried."
)


def reason(step, text):
    result = step.filter(Document(text, "0"))
    return result if result is True else result[1]


def removed_reasons(path):
    with gzip.open(path, "rt") as file:
        return [(doc["id"], doc["metadata"]["filter_reason"]) for doc in map(json.loads, file)]


def test_split_indic_words_follows_indicnlp_rules():
    assert split_indic_words("ధర: 1,000 రూపాయలు.") == ["ధర", ":", "1,000", "రూపాయలు", "."]
    assert split_indic_words('"అది" ‘ఈ’ 2020లో\nశీర్షిక') == ['"', "అది", '"', "‘ఈ’", "2020లో\nశీర్షిక"]


def test_ends_sentence():
    assert ends_sentence("అది ఉంది.  ")
    assert ends_sentence("ఇది ఏమిటి?")
    assert ends_sentence("यह है।")
    assert not ends_sentence("శీర్షిక")
    assert not ends_sentence('he said "stop."')


def test_repetition_helpers():
    assert find_duplicates(["a", "b", "a", "a"]) == (2, 2)
    assert top_ngram_chars(["a", "b", "a", "b"], 2) == 6
    assert duplicated_ngram_chars(["x", "y", "x", "y"], 2) == 2
    assert duplicated_ngram_chars(["ab", "c", "a", "bc"], 2) == 0


def test_gopher_repetition_reasons():
    step = GopherRepetitionFilter()
    assert reason(step, ENGLISH) is True
    assert reason(step, "same line\nsame line\nsame line\nother line") == "dup_line_frac"
    assert reason(step, "a paragraph\n\na paragraph\n\nanother one") == "dup_para_frac"
    assert reason(step, "buy now " * 20) == "top_2_gram"
    assert reason(step, "Rivers carry water downhill.") == "top_2_gram"
    assert reason(step, " \n ") == "empty"


def test_fineweb_quality_reasons():
    step = FineWebQualityFilter()
    assert reason(step, ENGLISH) is True
    assert reason(step, "a heading without punctuation\nanother heading here") == "line_punct_ratio"
    assert reason(step, "Short.\nTiny.\nSmall.") == "short_line_ratio"
    assert reason(step, "This exact line repeats several times in the document.\n" * 3) == "char_dup_ratio"
    listy = FineWebQualityFilter(short_line_thr=None, char_duplicates_ratio=None)
    assert reason(listy, "\n".join(f"{i}." for i in range(10))) == "list_ratio"
    assert reason(step, "\n\n") == "empty"


def test_telugu_preset_keeps_clean_telugu_and_drops_repetition():
    filters = telugu_quality_filters()
    assert all(reason(step, VILLAGE) is True for step in filters)
    spam = "\n".join(["ఇప్పుడే కొనండి, ఉచిత బహుమతి!"] * 5 + [VILLAGE])
    assert reason(filters[0], spam) == "dup_line_frac"


def test_telugu_preset_writes_removed_documents_per_filter(tmp_path):
    docs = [Document(VILLAGE, "kept"), Document("ఒక పదం", "tiny"), Document(ENGLISH, "english")]
    data = iter(docs)
    for step in telugu_quality_filters(removed_folder=str(tmp_path)):
        data = step.run(data)
    assert [doc.id for doc in data] == ["kept"]
    assert removed_reasons(tmp_path / "gopher_repetition/00000.jsonl.gz") == [("tiny", "top_2_gram")]
    assert removed_reasons(tmp_path / "gopher_quality/00000.jsonl.gz") == [("english", "gopher_not_enough_stop_words")]
