from io import BytesIO

import pyarrow.parquet as pq
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from dataflow.pipeline.filters import LanguageFilter, URLFilter
from dataflow.recipes.english_web import english_web, local

ARTICLES = {
    "river": [
        "The river runs through the old town, and people have built bridges of stone and wood across it.",
        "Every spring the water rises to the edge of the market, so traders move their stalls up the hill.",
        "Children learn to swim in the shallow parts while fishermen point out the currents to avoid.",
        "After a storm the river is loud, but the town still feels calm to the people who live there.",
        "A small museum by the bank keeps maps that show how the river has changed course over centuries.",
    ],
    "mountain": [
        "Climbers usually start before dawn so that they reach the summit while the weather is clear.",
        "The path above the tree line is steep, and a rope helps on the last rocky section near the top.",
        "Shepherds have used the high meadows for generations, moving their flocks up as the snow melts.",
        "From the ridge you can see three valleys and, on a good day, the distant shine of the sea.",
        "Rescue teams ask every hiker to leave a note of their route at the hut before setting out.",
    ],
    "harbour": [
        "Fishing boats return to the harbour each morning with crates of mackerel, crab and sardines.",
        "The lighthouse at the end of the pier was rebuilt after a winter gale damaged its old lamp.",
        "Ferries leave twice a day for the islands, and tickets sell out quickly during the summer months.",
        "A market on the quay sells fresh catch, nets, rope and hot tea to the crews coming ashore.",
        "Engineers are dredging the channel so that larger cargo ships can dock without waiting for tide.",
    ],
    "forest": [
        "Oak, beech and birch trees cover the hills, and their leaves turn gold and red in October.",
        "Rangers count the deer every winter and repair the fences that keep them out of young plantings.",
        "Mushroom pickers follow strict rules so that the rarest species can recover from year to year.",
        "A wooden tower lets visitors watch owls and woodpeckers without disturbing the nesting birds.",
        "Volunteers clear the old charcoal paths so that families can walk safely through the woods.",
    ],
    "valley": [
        "The valley school opened a new library this year, funded by a bake sale and a winter concert.",
        "Farmers grow apples, pears and plums on the sunny slopes and press cider in the autumn.",
        "A narrow railway connects the villages, and its small blue trains run every hour until evening.",
        "Readers send their news, recipes and photographs, and the paper prints a selection every week.",
        "Last month the valley council voted to restore the stone bridge that floods every spring.",
    ],
}


def article(topic: str, extra: str = "") -> bytes:
    paragraphs = "".join(f"<p>{sentence}</p>" for sentence in ARTICLES[topic]) + (f"<p>{extra}</p>" if extra else "")
    return f"<html><body><article><h1>About the {topic}</h1>{paragraphs}</article></body></html>".encode()


def write_warc(path, pages):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        writer = WARCWriter(file, gzip=True)
        for url, body in pages:
            http = StatusAndHeaders("200 OK", [("Content-Type", "text/html")], protocol="HTTP/1.1")
            writer.write_record(writer.create_warc_record(url, "response", payload=BytesIO(body), http_headers=http))


class AlwaysEnglish:
    def predict(self, text):
        return {"en": 0.99}


def test_the_recipe_filters_deduplicates_and_masks(tmp_path):
    pages = [
        (f"https://{topic}.example.com/story", article(topic)) for topic in ("river", "mountain", "harbour", "forest")
    ]
    pages.append(("https://copy.example.org/story", article("river")))
    pages.append(("https://casinobonus.example.net/story", article("forest")))
    pages.append(("https://short.example.com/", b"<html><body><p>Too short.</p></body></html>"))
    pages.append(
        (
            "https://contact.example.com/",
            article("valley", "Send your letters to the editor at editor@valley-news.com before Friday."),
        )
    )
    write_warc(tmp_path / "warc/a.warc.gz", pages)

    stats = english_web(
        str(tmp_path / "warc"),
        str(tmp_path / "out"),
        executor=local(workers=1),
        url_filter=URLFilter(ut1_categories=(), banned_subwords={"casinobonus"}),
        language_filter=LanguageFilter(["en"], threshold=0.65, lid=AlwaysEnglish()),
    )

    assert list(stats) == ["filter", "signatures", "buckets", "clusters", "dedup"]
    rows = pq.read_table(tmp_path / "out/output").to_pylist()
    urls = sorted(row["url"] for row in rows)
    assert len(urls) == 5
    assert "https://casinobonus.example.net/story" not in urls
    assert "https://short.example.com/" not in urls
    assert ("https://copy.example.org/story" in urls) != ("https://river.example.com/story" in urls)
    contact = next(row for row in rows if row["url"] == "https://contact.example.com/")
    assert "editor@valley-news.com" not in contact["text"]
    assert list((tmp_path / "out/removed/7_minhash").rglob("*.jsonl.gz"))
