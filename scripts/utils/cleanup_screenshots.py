"""
Clean up old test screenshots, keeping only match captures.
"""

from pathlib import Path
import os


def cleanup_screenshots():
    """Remove test screenshots, keep match captures."""
    screenshots_dir = Path("screenshots")

    # Files to keep (match captures)
    keep_patterns = ["match_start_*.png", "match_end_*.png"]

    # Get all PNG files
    all_files = list(screenshots_dir.glob("*.png"))

    # Find files to keep
    keep_files = set()
    for pattern in keep_patterns:
        keep_files.update(screenshots_dir.glob(pattern))

    # Find files to delete
    delete_files = [f for f in all_files if f not in keep_files]

    if not delete_files:
        print("No test files to clean up!")
        return

    print(f"Found {len(delete_files)} test files to clean up:")
    for f in delete_files:
        print(f"  - {f.name}")

    confirm = input(f"\nDelete these {len(delete_files)} files? (y/n): ").strip().lower()

    if confirm == 'y':
        for f in delete_files:
            f.unlink()
            print(f"Deleted: {f.name}")
        print(f"\nCleanup complete! Deleted {len(delete_files)} files.")
    else:
        print("Cleanup cancelled.")

    # Show remaining files
    remaining = list(screenshots_dir.glob("*.png"))
    print(f"\nRemaining screenshots: {len(remaining)}")
    for f in sorted(remaining):
        print(f"  - {f.name}")


if __name__ == "__main__":
    cleanup_screenshots()
