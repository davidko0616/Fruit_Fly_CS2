"""Timestamp matching for read-only CS2 perception streams."""
from bisect import bisect_left, bisect_right


class TimestampMatcher:
    """Match monotonically ordered rows to the nearest row within an age limit."""

    def __init__(self, rows, time_key):
        self.rows = list(rows)
        self.time_key = time_key
        self.timestamps = [int(row[time_key]) for row in self.rows]
        if any(timestamp < 0 for timestamp in self.timestamps):
            raise ValueError(f'{time_key} must be nonnegative')
        if any(right <= left for left, right in zip(
                self.timestamps, self.timestamps[1:])):
            raise ValueError(f'{time_key} must be strictly increasing')

    def nearest(self, timestamp, max_delta_ns):
        timestamp = int(timestamp)
        max_delta_ns = int(max_delta_ns)
        if timestamp < 0 or max_delta_ns < 0:
            raise ValueError('timestamp and max_delta_ns must be nonnegative')
        if not self.rows:
            return None, None
        insertion = bisect_left(self.timestamps, timestamp)
        candidates = []
        if insertion:
            candidates.append(insertion - 1)
        if insertion < len(self.rows):
            candidates.append(insertion)
        index = min(candidates, key=lambda candidate: (
            abs(self.timestamps[candidate] - timestamp),
            self.timestamps[candidate] > timestamp))
        delta_ns = self.timestamps[index] - timestamp
        if abs(delta_ns) > max_delta_ns:
            return None, delta_ns
        return self.rows[index], delta_ns

    def latest(self, timestamp, max_age_ns):
        """Return the latest causal row at or before timestamp."""
        timestamp = int(timestamp)
        max_age_ns = int(max_age_ns)
        if timestamp < 0 or max_age_ns < 0:
            raise ValueError('timestamp and max_age_ns must be nonnegative')
        insertion = bisect_right(self.timestamps, timestamp)
        if not insertion:
            return None, None
        index = insertion - 1
        delta_ns = self.timestamps[index] - timestamp
        if -delta_ns > max_age_ns:
            return None, delta_ns
        return self.rows[index], delta_ns
