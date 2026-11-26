from __future__ import annotations

import argparse
import ctypes
import importlib.util
import math
import os
import shutil
import sys
import textwrap
from pathlib import Path

ML_CLI_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_CLI_DIR / "sentiment_model.joblib"
PREPROCESSING_PATH = ML_CLI_DIR / "preprocessing.py"
PREPROCESSING_MODULE_NAME = "ml_cli_sentiment_preprocessing"
SENTIMENT_LABELS = ("negative", "neutral", "positive")

try:
    import joblib
    import nltk
    import sys as _sys
    if PREPROCESSING_MODULE_NAME not in _sys.modules:
        spec = importlib.util.spec_from_file_location(
            PREPROCESSING_MODULE_NAME, PREPROCESSING_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load preprocessing module at {PREPROCESSING_PATH}")
        preprocessing = importlib.util.module_from_spec(spec)
        _sys.modules[PREPROCESSING_MODULE_NAME] = preprocessing
        spec.loader.exec_module(preprocessing)
    else:
        preprocessing = _sys.modules[PREPROCESSING_MODULE_NAME]
    tokenize_for_sentiment = preprocessing.tokenize_for_sentiment
except ImportError as exc:
    raise SystemExit(
        "The sentiment CLI dependencies are missing. Run it with the project's "
        ".venv313 environment."
    ) from exc


class TerminalUI:
    """Colored terminal output with a plain-text fallback."""

    PALETTE = {
        "accent": (115, 168, 255),
        "text": (236, 238, 241),
        "muted": (150, 164, 184),
        "rule": (64, 78, 102),
        "negative": (239, 88, 88),
        "positive": (92, 190, 112),
        "neutral": (242, 164, 65),
        "error": (255, 105, 105),
    }

    def __init__(self) -> None:
        self.use_color = "NO_COLOR" not in os.environ and sys.stdout.isatty()
        self.width = max(40, min(shutil.get_terminal_size((80, 24)).columns - 4, 72))
        if self.use_color:
            self._enable_windows_ansi()

    @staticmethod
    def _enable_windows_ansi() -> None:
        if os.name != "nt":
            return
        try:
            kernel32 = ctypes.windll.kernel32
            stdout_handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(stdout_handle, mode.value | 0x0004)
        except (AttributeError, OSError, TypeError):
            pass

    def paint(self, value: str, tone: str = "text", bold: bool = False) -> str:
        if not self.use_color:
            return value
        red, green, blue = self.PALETTE[tone]
        weight = "1;" if bold else ""
        return f"\033[{weight}38;2;{red};{green};{blue}m{value}\033[0m"

    def _box_row(self, value: str, tone: str = "text", bold: bool = False) -> None:
        content_width = self.width - 3
        shown = value[:content_width]
        padding = " " * max(0, content_width - len(shown))
        print(f"  \u2502 {self.paint(shown, tone, bold)}{padding}\u2502")

    def banner(self) -> None:
        rule = self.paint("\u2500" * (self.width - 2), "rule")
        print(f"  \u256d{rule}\u256e")
        self._box_row("TWEET SENTIMENT", "accent", bold=True)
        self._box_row("NEGATIVE  \u00b7  NEUTRAL  \u00b7  POSITIVE", "muted")
        print(f"  \u2570{rule}\u256f\n")

    def show_tweet(self, tweet: str) -> None:
        print(self.paint("  YOU", "muted", bold=True))
        display_text = " ".join(tweet.split())
        wrap_width = max(28, self.width - 8)
        marker = self.paint("\u203a", "accent", bold=True)
        for line in textwrap.wrap(display_text, width=wrap_width) or [""]:
            print(f"  {marker} {line}")
        print()

    def show_results(self, results: list[tuple[str, float]]) -> None:
        sentiment, score = results[0]
        print(self.paint("  PREDICTION", "muted", bold=True))
        marker = self.paint("\u25c6", sentiment, bold=True)
        sentiment_name = self.paint(sentiment.upper(), sentiment, bold=True)
        estimate = self.paint(f"{score:.1f}% strongest estimate", "text")
        print(f"  {marker} {sentiment_name}  {estimate}\n")
        print(self.paint("  CLASS DISTRIBUTION", "muted", bold=True))

        columns = shutil.get_terminal_size((80, 24)).columns
        bar_width = max(10, min(24, columns - 48))
        for label, percentage in results:
            filled = max(0, min(bar_width, round(percentage / 100 * bar_width)))
            empty = bar_width - filled
            bar = self.paint("\u2501" * filled, label, bold=True)
            bar += self.paint("\u2500" * empty, "rule")
            print(f"  {label.title():<10} {bar}  {percentage:5.1f}%")
        print()
        print(self.paint("  Scores are model estimates, not certainty.", "muted"))
        print()

    def error(self, message: str) -> None:
        marker = self.paint("!", "error", bold=True)
        print(f"\n  {marker} {message}", file=sys.stderr)


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Selected classical model not found at {MODEL_PATH}. "
            "Run Sentiment-Analysis.ipynb through its model export cell first."
        )
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Could not load the saved model: {exc}") from exc

    required = {"model", "classes", "sentiment_labels", "token_column", "model_name"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"The saved model bundle is missing: {', '.join(sorted(missing))}")
    if set(bundle["sentiment_labels"].values()) != set(SENTIMENT_LABELS):
        raise ValueError("The saved model does not have the expected three sentiment labels.")
    return bundle


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def predict_sentiments(text: str, bundle: dict) -> list[tuple[str, float]]:
    try:
        tokens = tokenize_for_sentiment(text, bundle["token_column"])
    except LookupError as exc:
        raise SystemExit(
            "NLTK data is missing. Run the NLTK setup cell in Sentiment-Analysis.ipynb first."
        ) from exc

    model = bundle["model"]
    class_ids = [int(class_id) for class_id in bundle["classes"]]
    if hasattr(model, "predict_proba"):
        scores = [float(score) for score in model.predict_proba([tokens])[0]]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function([tokens])
        if getattr(decision, "ndim", 1) > 1:
            decision = decision[0]
        scores = _softmax([float(score) for score in decision])
    else:
        predicted = int(model.predict([tokens])[0])
        scores = [1.0 if class_id == predicted else 0.0 for class_id in class_ids]

    labels = {
        int(class_id): str(label)
        for class_id, label in bundle["sentiment_labels"].items()
    }
    results = [
        (labels[class_id], score * 100.0)
        for class_id, score in zip(class_ids, scores)
    ]
    return sorted(results, key=lambda item: item[1], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate negative, neutral, or positive sentiment in a tweet."
    )
    parser.add_argument(
        "tweet", nargs="*", help="Tweet text. Omit it for interactive mode."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ui = TerminalUI()
    ui.banner()

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        ui.error(str(exc))
        return 1

    print(f"  Model: {bundle['model_name']}\n")

    if args.tweet:
        tweet = " ".join(args.tweet).strip()
        if not tweet:
            ui.error("Please enter a non-empty tweet.")
            return 2
        try:
            ui.show_tweet(tweet)
            ui.show_results(predict_sentiments(tweet, bundle))
        except SystemExit as exc:
            ui.error(str(exc))
            return 1
        except Exception as exc:
            ui.error(f"Could not classify this tweet: {exc}")
            return 1
        return 0

    print("  Type a tweet to analyze. Use /help for commands or /exit to quit.\n")
    while True:
        try:
            prompt = ui.paint("  ? ", "accent", bold=True)
            tweet = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  " + ui.paint("See you next time.", "muted"))
            return 0

        if tweet.lower() in {"/exit", "/quit"}:
            print("\n  " + ui.paint("See you next time.", "muted"))
            return 0
        if tweet.lower() == "/help":
            print("\n  Enter a tweet to see its sentiment estimates.")
            print("  /help  show this message")
            print("  /exit  close the session\n")
            continue
        if not tweet:
            continue

        try:
            ui.show_tweet(tweet)
            ui.show_results(predict_sentiments(tweet, bundle))
        except SystemExit as exc:
            ui.error(str(exc))
        except Exception as exc:
            ui.error(f"Could not classify this tweet: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
