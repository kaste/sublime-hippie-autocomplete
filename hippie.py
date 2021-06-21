import sublime
import sublime_plugin
from collections import defaultdict
from itertools import chain
import re

flatten = chain.from_iterable
VIEW_TOO_BIG = 1000000
WORD_PATTERN = re.compile(r'(\w{2,})', re.S)  # Start from words of length 2

index = {}  # type: Dict[sublime.View, Tuple[int, Set[str]]]
last_view = None
initial_primer = ""
matching = []
last_index = 0
history = defaultdict(dict)  # type: Dict[sublime.Window, Dict[str, str]]


class HippieWordCompletionCommand(sublime_plugin.TextCommand):
    def run(self, edit):
        global last_view, matching, last_index, initial_primer
        window = self.view.window()
        assert window

        first_sel = self.view.sel()[0]
        word_region = self.view.word(first_sel)
        primer_region = sublime.Region(word_region.a, first_sel.end())
        primer = self.view.substr(primer_region)

        def _matching(primer, exclude):
            if primer in history[window]:
                yield history[window][primer]
            yield from fuzzyfind(primer, index_for_view(self.view) - exclude)
            yield from fuzzyfind(
                primer,
                index_for_other_views(self.view) - index_for_view(self.view)
            )
            yield primer  # Always be able to cycle back

        if last_view is not self.view or not matching or primer != matching[last_index]:
            word_under_cursor = (
                primer
                if word_region == primer_region
                else self.view.substr(word_region)
            )
            last_view = self.view
            initial_primer = primer
            matching = ldistinct(_matching(
                initial_primer,
                exclude={word_under_cursor}
            ))
            last_index = 0

        if matching[last_index] == primer:
            last_index += 1
        if last_index >= len(matching):
            last_index = 0

        for region in self.view.sel():
            self.view.replace(
                edit,
                sublime.Region(self.view.word(region).a, region.end()),
                matching[last_index]
            )

        history[window][initial_primer] = matching[last_index]


class HippieListener(sublime_plugin.EventListener):
    def on_init(self, views):
        for view in views:
            index_for_view(view)

    def on_close(self, view):
        global index
        index.pop(view, None)



def index_for_view(view):
    global index
    change_count = view.change_count()
    try:
        _change_count, words = index[view]
        if _change_count != change_count:
            raise KeyError("view has changed")
    except KeyError:
        words = _index_view(view)
        index[view] = (change_count, words)
    return words


def _index_view(view):
    if view.size() > VIEW_TOO_BIG:
        return set()
    contents = view.substr(sublime.Region(0, view.size()))
    return set(WORD_PATTERN.findall(contents))


def index_for_other_views(view):
    return set(flatten(index_for_view(v) for v in other_views(view)))


def other_views(view):
    return (v for v in view.window().views() if v != view)


def fuzzyfind(primer, collection, sort_results=True):
    """
    Args:
        primer (str): A partial string which is typically entered by a user.
        collection (iterable): A collection of strings which will be filtered
                               based on the `primer`.
        sort_results(bool): The suggestions are sorted by considering the
                            smallest contiguous match, followed by where the
                            match is found in the full string. If two suggestions
                            have the same rank, they are then sorted
                            alpha-numerically. This parameter controls the
                            *last tie-breaker-alpha-numeric sorting*. The sorting
                            based on match length and position will be intact.
    Returns:
        suggestions (generator): A generator object that produces a list of
            suggestions narrowed down from `collection` using the `primer`.
    """
    suggestions = []
    for item in collection:
        if score := fuzzy_score(primer.lower(), item):
            suggestions.append((score, item))

    if sort_results:
        return [z[-1] for z in sorted(suggestions)]
    else:
        return [z[-1] for z in sorted(suggestions, key=lambda x: x[0])]


def fuzzy_score(primer, item):
    start, pos, prev, score = -1, -1, 0, 1
    item_l = item.lower()
    for c in primer:
        pos = item_l.find(c, pos + 1)
        if pos == -1:
            return
        if start == -1:
            start = pos

        # Update score if not at start of the word
        if pos > 0:
            pc = item[pos - 1]
            if pc in "_-" or pc.isupper() < item[pos].isupper():
                continue
        score += pos - prev
        prev = pos

    return (score, len(item))


def ldistinct(seq):
    """Iterates over sequence skipping duplicates"""
    seen = set()
    res = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            res.append(item)
    return res
