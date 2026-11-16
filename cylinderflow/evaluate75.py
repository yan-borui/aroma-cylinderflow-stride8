"""Score native 1+74 predictions on Validation75 or official Test75."""

from .rollout75 import main


if __name__ == "__main__":
    main(test_frames=75)
