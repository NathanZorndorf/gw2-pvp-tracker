import pytest
from collections import defaultdict
import math

# Stash key for accumulating stats safely across tests in a single session
stats_key = pytest.StashKey()

def pytest_configure(config):
    # Initialize the stash with an empty list
    config.stash[stats_key] = []

def pytest_addoption(parser):
    parser.addoption(
        "--show-stats",
        action="store_true",
        default=False,
        help="Print detailed accuracy statistics summary at the end of the test session."
    )

@pytest.fixture(scope="session")
def stats_recorder(request):
    """
    Fixture to record statistics from tests.
    Usage in test:
        stats_recorder.append({
            'test': 'test_name',
            'category': 'Icon Matching',
            'correct': 10,
            'total': 12,
            'details': 'failed on img1.png'
        })
    """
    return request.config.stash[stats_key]

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--show-stats"):
        return

    stats = config.stash[stats_key]
    if not stats:
        terminalreporter.section("Accuracy Statistics", sep="=")
        terminalreporter.write_line("No statistics recorded (did you add stats_recorder to your tests?)")
        return

    terminalreporter.section("Accuracy Statistics Summary", sep="=")
    
    # Aggregate stats by category
    aggregated = defaultdict(lambda: {'correct': 0, 'total': 0, 'tests': []})
    
    for entry in stats:
        cat = entry.get('category', 'General')
        aggregated[cat]['correct'] += entry.get('correct', 0)
        aggregated[cat]['total'] += entry.get('total', 0)
        aggregated[cat]['tests'].append(entry)

    # Print summary table
    terminalreporter.write_line(f"{'Category':<30} | {'Accuracy':<10} | {'Correct':<8} | {'Total':<8}")
    terminalreporter.write_line("-" * 65)
    
    for cat, data in sorted(aggregated.items()):
        correct = data['correct']
        total = data['total']
        accuracy = (correct / total * 100) if total > 0 else 0.0
        terminalreporter.write_line(f"{cat:<30} | {accuracy:>9.1f}% | {correct:<8} | {total:<8}")

    terminalreporter.write_line("-" * 65)
    total_correct = sum(d['correct'] for d in aggregated.values())
    total_total = sum(d['total'] for d in aggregated.values())
    total_acc = (total_correct / total_total * 100) if total_total > 0 else 0.0
    terminalreporter.write_line(f"{'OVERALL':<30} | {total_acc:>9.1f}% | {total_correct:<8} | {total_total:<8}")
    
    terminalreporter.write_line("")
