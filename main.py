from spam_classifier.gui import launch_gui
from spam_classifier.SpamEmailClassifier import default_dataset_path


def main() -> int:
    launch_gui(default_dataset=default_dataset_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
