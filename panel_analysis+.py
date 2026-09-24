"""Run the panel analysis with additional tests using median-split age."""

from panel_analysis import main


if __name__ == "__main__":
    main(include_age_split=True)
