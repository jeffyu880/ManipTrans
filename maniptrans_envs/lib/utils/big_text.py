"""Big ASCII digit rendering for terminal countdowns (P-key reset, SPACE start)."""

# Seven-segment layout used to draw the big countdown digits in the terminal.
#   segments:  a=top  b=upper-right  c=lower-right  d=bottom  e=lower-left  f=upper-left  g=middle
_SEVEN_SEG_DIGITS = {
    "0": "abcdef",
    "1": "bc",
    "2": "abdeg",
    "3": "abcdg",
    "4": "bcfg",
    "5": "acdfg",
    "6": "acdefg",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
}


def render_big_number(value, seg_width=12, seg_height=3):
    """Render an integer as big, thick ASCII digits from '-' and '|'.

    Bars are 2 chars thick (horizontals 2 rows, verticals 2 columns); each digit is
    `6 + 2*seg_height` lines tall.
    """
    stroke = 2
    full_width = stroke + seg_width + stroke
    lines_per_digit = []
    for char in str(value):
        segments = _SEVEN_SEG_DIGITS[char]

        def horizontal(is_on):
            return [("-" if is_on else " ") * full_width] * stroke

        def vertical(left_on, right_on):
            row = ("|" if left_on else " ") * stroke + " " * seg_width + ("|" if right_on else " ") * stroke
            return [row] * seg_height

        rows = []
        rows += horizontal("a" in segments)                    # top
        rows += vertical("f" in segments, "b" in segments)     # upper verticals
        rows += horizontal("g" in segments)                    # middle
        rows += vertical("e" in segments, "c" in segments)     # lower verticals
        rows += horizontal("d" in segments)                    # bottom
        lines_per_digit.append(rows)

    # Join digits side by side with a two-space gutter.
    return "\n".join("  ".join(digit_rows[row] for digit_rows in lines_per_digit) for row in range(len(lines_per_digit[0])))
