"""Caller-owned, resumable enumeration of bounded physical macro recipes."""
import time

MACRO_EXPANSION = object()
MACRO_PAUSE = object()
_UNSTARTED = object()


class MacroContinuation:
    """Keep a recipe iterator and its exact in-flight physical state alive.

    Iterators yield ``MACRO_EXPANSION`` before beginning each counted recipe,
    and ``None`` at bounded physical checkpoints. They update ``results`` only
    with completely simulated arrivals. The runner owns expansion accounting;
    the iterator owns simulated-frame accounting and ranking its result batch.
    """
    def __init__(self, node_budget=None):
        self.key = None
        self.work = None
        self.pending = _UNSTARTED
        self.done = False
        self.expansions = 0
        self.resumes = 0
        self.results = []
        self.children = []
        self.node_budget = node_budget
        self.recipe_active = False

    def advance(self, key, factory, *, deadline, stats, max_expansions=None,
                recipe_limit=None):
        if self.key is None:
            self.key = key
            self.work = iter(factory(self))
        elif self.key != key:
            raise ValueError("Macro continuation belongs to another physical edge")
        elif not self.done:
            self.resumes += 1
        while not self.done and (time.monotonic() < deadline or
                self.recipe_active and self.pending is not MACRO_PAUSE
                and max_expansions is not None and stats.get("expansions",0)>=max_expansions):
            if self.pending is MACRO_EXPANSION:
                if recipe_limit is not None and self.expansions >= recipe_limit:
                    self._finish()
                    break
                if max_expansions is not None and stats.get("expansions", 0) >= max_expansions:
                    break
                stats["expansions"] = stats.get("expansions", 0) + 1
                self.expansions += 1
                self.recipe_active = True
            try:
                before = stats.get("expansions", 0)
                self.pending = next(self.work)
            except StopIteration as finished:
                if finished.value is not None:
                    self.results = finished.value
                self._finish()
            finally:
                # Nested relay searches own their graph-node accounting, but
                # still consume this macro family's cumulative work allowance.
                self.expansions += stats.get("expansions", 0) - before
            if self.pending is MACRO_EXPANSION:
                # If the last available expansion was already paid, finish
                # that bounded recipe even when its final checkpoint crosses
                # the time slice. Never begin the next recipe during this
                # drain. Otherwise the outer global budget guard would strand
                # the in-flight action sequence permanently.
                self.recipe_active = False
            if self.pending is MACRO_PAUSE:
                break
        return list(self.results)

    def _finish(self):
        self.done = True
        if self.work is not None:
            self.work.close()
            self.work = None
        for child in self.children:
            child.release()
        self.children.clear()

    def release(self):
        self._finish()
        self.results.clear()
