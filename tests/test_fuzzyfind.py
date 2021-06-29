from pathlib import Path
import sys

from unittesting import DeferrableTestCase
from .parameterized import parameterized as p

this_package_name = Path(__file__).parent.parent.stem
fuzzyfind = sys.modules[f"{this_package_name}.hippie"].fuzzyfind


class xmark(str):
    expected_failure = True


def x(a, *rest):
    return (xmark(a), *rest)


class TestFuzyyFind(DeferrableTestCase):

    @p.expand([
        ("el", ["else", "EventListener"], "else"),
        ("EL", ["else", "EventListener"], "EventListener"),
        ("eL", ["else", "EventListener"], "EventListener"),
        ("eL", ["eventListener", "EventListener"], "eventListener"),

        ("fbs", ["find_by", "fibs_by"], "fibs_by"),
        ("fbs", ["fibs_by", "fibs_bs"], "fibs_bs"),

        ("prre", ["primer_region", "primere_region"], "primer_region"),

        ("win", ["window", "Window"], "window"),
        ("Win", ["window", "Window"], "Window"),

        ("rn", ["the_foo", "remote_name"], "remote_name"),
        ("rn", ["foorn_zoo", "remote_name"], "remote_name"),

        ("rn", ["rnmote_na", "remote_name"], "rnmote_na"),
       x("rn", ["rnmote_name", "ramote_name"], "rnmote_name"),
        ("rn", ["rnmote_name", "rxmote_name"], "rnmote_name"),

        ("rn", ["foor_nzoo", "remote_name"], "remote_name"),
        ("rn", ["foo_rnzoo", "remote_name"], "remote_name"),

        ("name", ["abc_name", "name_abc"], "name_abc"),
        ("name", ["abc_def_name", "name_abc"], "name_abc"),

        ("the", ["The", "the"], "the"),
        ("teh", ["The", "the"], "the"),
        ("teh", ["they", "the"], "the"),
        ("teh", ["they", "The"], "they"),


    ])
    def test_fuzzyfind(self, primer, words, best_completion):
        if isinstance(primer, xmark):
            self.assertRaises(
                AssertionError,
                lambda: self.assertEqual(fuzzyfind(primer, words)[0], best_completion)
            )
        else:
            self.assertEqual(fuzzyfind(primer, words)[0], best_completion)
