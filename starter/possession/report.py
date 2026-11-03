"""The per-half tally printed after a run (restarts found, stoppages, vetoes ...)."""


class Report:
    """Counts in the order they were first seen (the printed order). kickoff is the kickoff frame."""

    def __init__(self):
        self.kickoff = None
        self.counts = {}

    def add(self, key, n=1):
        self.counts[key] = self.counts.get(key, 0) + n

    def set(self, key, n):
        self.counts[key] = n

    def remove(self, key):
        """Take one back; a count that reaches zero disappears."""
        self.counts[key] -= 1
        if not self.counts[key]:
            del self.counts[key]
