def normalize_index(index: int, length: int) -> int:
    if index < 0:
        index += length

    if not 0 <= index < length:
        raise IndexError(index)

    return index


def normalize_range(start: int, end: int, length: int) -> tuple[int, int]:
    if start < 0:
        start += length

    if end < 0:
        end += length

    if not 0 <= start <= end <= length:
        raise IndexError(f"Invalid range [{start}:{end}]")

    return start, end