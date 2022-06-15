def log_progress_bar(percent=0.0, width=40):
    left = int(width * percent)
    right = width - left

    tags = "#" * left
    spaces = " " * right
    percents = f"{(percent * 100):.0f}%"

    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)
