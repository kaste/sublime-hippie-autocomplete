import sublime
import sublime_plugin
from collections import defaultdict
from itertools import chain, cycle
from pathlib import Path
import re


from typing import (
    Dict, Generic, Iterable, Iterator, List, Optional, Set, Tuple, TypeVar
)
T = TypeVar("T")


flatten = chain.from_iterable
VIEW_TOO_BIG = 1000000

index = {}  # type: Dict[sublime.View, Tuple[int, Set[str]]]
last_view = None  # type: Optional[sublime.View]
initial_primer = ""  # type: str
matching = None  # type: Optional[back_n_forth_iterator[str]]
last_suggestion = None  # type: Optional[str]
history = defaultdict(dict)  # type: Dict[sublime.Window, Dict[str, str]]


def plugin_loaded():
    install_low_priority_package()


def plugin_unloaded():
    try:
        import package_control
    except ImportError:
        pass
    else:
        this_package_name = Path(__file__).parent.stem
        if package_control.events.remove(this_package_name):
            uninstall_low_priority_package()


from contextlib import contextmanager
import time
import threading


@contextmanager
def print_runtime(message):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = round((end_time - start_time) * 1000)
    thread_name = threading.current_thread().name[0]
    print('{} took {}ms [{}]'.format(message, duration, thread_name))


def get_primer(view: sublime.View) -> str:
    first_sel = view.sel()[0]
    word_region = view.word(first_sel)
    primer_region = sublime.Region(word_region.a, first_sel.end())
    return view.substr(primer_region)


class HippieWordCompletionCommand(sublime_plugin.TextCommand):
    @print_runtime("completion")
    def run(self, edit, forwards=True) -> None:
        global last_view, matching, initial_primer, last_suggestion
        window = self.view.window()
        assert window

        first_sel = self.view.sel()[0]
        word_region = self.view.word(first_sel)
        primer_region = sublime.Region(word_region.a, first_sel.end())
        primer = self.view.substr(primer_region)

        def _matching(primer, exclude) -> Iterator[str]:
            assert window  # fix false mypy complain
            # Add `primer` at the front to allow going back to it, either
            # using `shift+tab` or when cycling through all possible
            # completions.
            yield primer
            if primer in history[window]:
                yield history[window][primer]
            views_index = index_for_view(self.view)
            yield from fuzzyfind(primer, views_index - exclude)
            yield from fuzzyfind(
                primer, index_for_other_views(self.view) - views_index
            )

        if last_view is not self.view or not matching or primer != last_suggestion:
            word_under_cursor = (
                primer
                if word_region == primer_region
                else self.view.substr(word_region)
            )
            last_view = self.view
            initial_primer = primer
            matching = back_n_forth_iterator(cycle(unique_everseen(_matching(
                initial_primer,
                exclude={word_under_cursor}
            ))))
            next(matching)  # skip the `primer` we added at the front

        if forwards:
            last_suggestion = next(matching)
        else:
            try:
                last_suggestion = matching.prev()
            except ValueError:
                window.status_message("at first suggestions")
                return

        for region in self.view.sel():
            self.view.replace(
                edit,
                sublime.Region(self.view.word(region).a, region.end()),
                last_suggestion
            )

        if last_suggestion == initial_primer:
            history[window].pop(initial_primer, None)
        else:
            history[window][initial_primer] = last_suggestion


class HippieListener(sublime_plugin.EventListener):
    def on_init(self, views):
        for view in views:
            index_for_view(view)

    def on_close(self, view):
        global index
        index.pop(view, None)

    def on_pre_close_window(self, window):
        global history
        history.pop(window, None)

    def on_query_context(self, view, key, operator, operand, match_all) -> Optional[bool]:
        if key == "happy_hippie":
            if operator != sublime.OP_EQUAL:
                print(f"Context '{key}' only supports operator 'equal'.")
                return False

            if operand is not True:
                print(f"Context '{key}' only supports operand 'true'.")
                return False

            return self.happy_hippie_key(view, operator, operand, match_all)

        elif key == "just_hippie_completed":
            if operator != sublime.OP_EQUAL:
                print(f"Context '{key}' only supports operator 'equal'.")
                return False

            if operand is not True:
                print(f"Context '{key}' only supports operand 'true'.")
                return False

            return self.just_hippie_completed_key(view, operator, operand, match_all)

        return None

    def happy_hippie_key(self, view, operator, operand, match_all) -> bool:
        char_class = _get_char_class(view)
        re_cc = re.compile(r"{}+".format(char_class))
        for s in view.sel():
            if s.empty():
                if (
                    s.a == 0
                    or not re_cc.fullmatch(view.substr(s.a - 1))
                ):
                    # abort: previous char contains word_separators
                    return False
            else:
                if not re_cc.fullmatch(view.substr(s)):
                    # abort: selection contains word_separators
                    return False

        return True

    def just_hippie_completed_key(
        self, view, operator, operand, match_all
    ) -> bool:
        global last_view, matching, last_suggestion
        primer = get_primer(view)
        if view == last_view and matching and primer == last_suggestion:
            return True
        return False


def index_for_view(view: sublime.View) -> Set[str]:
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


def _index_view(view: sublime.View) -> Set[str]:
    if view.size() > VIEW_TOO_BIG:
        return set()
    contents = view.substr(sublime.Region(0, view.size()))
    char_class = _get_char_class(view)
    pattern = r"{}{{2,}}".format(char_class)
    return set(re.findall(pattern, contents))


def _get_char_class(view) -> str:
    word_separators = view.settings().get("word_separators", "")
    return r"[^\s{}]".format(
        word_separators.replace("\\", "\\\\").replace("]", r"\]")
    )


def index_for_other_views(view):
    return set(flatten(index_for_view(v) for v in other_views(view)))


def other_views(view):
    return (v for v in view.window().views() if v != view)


def fuzzyfind(primer: str, collection: Iterable[str], sort_results=True) -> List[str]:
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
        if score := fuzzy_score(primer, item):
            suggestions.append((score, item))

    if sort_results:
        return [z[-1] for z in sorted(suggestions)]
    else:
        return [z[-1] for z in sorted(suggestions, key=lambda x: x[0])]


def fuzzy_score(primer: str, item: str) -> Optional[Tuple[float, int]]:
    pos, score = -1, 0.0
    item_l = item.lower()
    primer_l = primer.lower()
    for idx in range(len(primer)):
        try:
            pos, _score = find_char(primer_l[idx:], item, item_l, pos + 1)
        except ValueError:
            return None

        score += 2 * _score
        if pos == 0:
            score -= 1
        if _score == 0 and primer[idx] == item[pos]:
            score -= 0.5 if primer[idx] == primer_l[idx] else 2

        if score > 10:
            return None

    return (score, len(item))


def find_char(primer_rest, item, item_l, start: int) -> Tuple[int, float]:
    prev = ''
    first_seen = -1
    needle = primer_rest[0]
    for idx, ch in enumerate(item[start:], start):
        if needle == ch.lower():
            if idx == start:
                return start, 0
            if first_seen == -1:
                first_seen = idx
            if ch.isupper() and (not prev or prev.islower()):
                return idx, 0
            if prev in "-_":
                return idx, 0
        prev = ch

    if first_seen == -1:
        if (
            start > 1
            and (pos := item_l.rfind(needle, 0, start - 1)) != -1
        ):
            return pos, start - 1 - pos
        raise ValueError(f"can't match {primer_rest!r} with {item!r}")
    if item.endswith(primer_rest):
        return len(item) - len(primer_rest), 1
    return first_seen, first_seen - (start - 1)


def unique_everseen(seq: Iterable[T]) -> Iterator[T]:
    """Iterates over sequence skipping duplicates"""
    seen = set()
    for item in seq:
        if item not in seen:
            seen.add(item)
            yield item


class back_n_forth_iterator(Generic[T]):
    def __init__(self, iterable: Iterable[T]) -> None:
        self._it = iter(iterable)  # type: Iterator[T]
        self._index = None  # type: Optional[int]
        self._cache = []  # type: List[T]

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self._index is not None:
            if self._index < len(self._cache) - 1:
                self._index += 1
                return self._cache[self._index]
            else:
                self._index = None

        val = next(self._it)
        self._cache.append(val)
        return val

    next == __next__

    def prev(self) -> T:
        if self._index is None:
            self._index = len(self._cache) - 1
        elif self._index <= 0:
            raise ValueError("can't rewind any further")
        self._index -= 1
        val = self._cache[self._index]
        return val


#  Install a *lowest* priority package for the key binding

import os
import zipfile
import zlib
package_name = "1_hippie_key_binding.sublime-package"


class HappyFileSyntax(sublime_plugin.EventListener):
    def on_load(self, view):
        if view.file_name() == __file__:
            this_package_name = Path(__file__).parent.stem
            syntax_file = f"Packages/{this_package_name}/hippie python.sublime-syntax"
            view.assign_syntax(syntax_file)


json = python = lambda s: s.lstrip()
keymap = json("""
[
    { "keys": ["tab"], "command": "hippie_word_completion",
      "context": [
        { "key": "read_only", "operator": "not_equal" },
        { "key": "auto_complete_visible", "operand": false },
        { "key": "has_snippet", "operand": false  },
        { "key": "has_next_field", "operand": false },
        { "key": "happy_hippie" },
    ]},
    { "keys": ["shift+tab"], "command": "hippie_word_completion",
      "args": {"forwards": false},
      "context": [
        { "key": "read_only", "operator": "not_equal" },
        { "key": "auto_complete_visible", "operand": false },
        { "key": "has_snippet", "operand": false  },
        { "key": "has_next_field", "operand": false },
        { "key": "just_hippie_completed" },
    ]}
]
""")

unloader = python("""
import os, sys
def plugin_loaded():
    if "{main_plugin}" not in sys.modules:
        os.remove(r"{package_fpath}")
        print("Uninstalled", r"{package_fpath}")
""")


def install_low_priority_package() -> None:
    this_plugin_name = Path(__file__).stem
    this_package_name = Path(__file__).parent.stem
    this_module = f"{this_package_name}.{this_plugin_name}"

    ipp = sublime.installed_packages_path()
    package_fpath = os.path.join(ipp, package_name)

    files_to_copy = {
        ".python-version": "3.8\n",
        "Default.sublime-keymap": keymap,
        "unloader.py": unloader.format(
            main_plugin=this_module, package_fpath=package_fpath
        )
    }

    if os.path.exists(package_fpath):
        with zipfile.ZipFile(package_fpath) as zfile:
            zipped_files = zfile.infolist()
            if files_to_copy.keys() == {f.filename for f in zipped_files}:
                for f in zipped_files:
                    contents = files_to_copy[f.filename]
                    if zlib.crc32(contents.encode()) != f.CRC:
                        break
                else:
                    return None

    def create_package():
        with zipfile.ZipFile(package_fpath, "w") as zfile:
            for target, source in files_to_copy.items():
                zfile.writestr(target, source)

        print("Installed", package_fpath)

    try:
        os.remove(package_fpath)
    except OSError:
        create_package()
    else:
        # wait for Sublime Text to unload the old package
        sublime.set_timeout(create_package, 1000)


def uninstall_low_priority_package() -> None:
    ipp = sublime.installed_packages_path()
    package_fpath = os.path.join(ipp, package_name)
    os.remove(package_fpath)
