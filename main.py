"""
hotel dynamic pricing - main pipeline
this runs everything from start to finish
data comes from real kaggle hotel bookings
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import run_preprocessing
from src.eda import run_eda
from src.models import run_modeling
from src.advanced_analysis import run_advanced_analysis

banner = """this is my attempt on the hotel pricing system
        i have taken the dataset from kaggle
        it contains about 120k records
        i then cleaned the data according to my needs
        i managed to get a high accuracy for all the models
        i chose the model which gave the highest accuracy
"""

def step(n, title):
    print(f"  step {n} | {title}")


def main():
    print(banner)
    t_total = time.time()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    step(1, "data loading")
    path = "data/hotel_booking_data.csv"

    if not os.path.exists(path):
        print("  raw dataset not found at the path")
        print("  place the hotel booking csv in the data/ folder and try again")
        sys.exit(1)

    import pandas as pd
    df_check = pd.read_csv(path, nrows=5)
    print(f"  dataset found, columns: {list(df_check.columns[:6])}")

    step(2, "data cleaning and preprocessing")

    if not os.path.exists("data/hotel_booking_clean.csv"):
        run_preprocessing()
    else:
        print("  clean dataset already exists, skipping")

    step(3, "exploratory data analysis (12 plots)")
    run_eda()

    step(4, "machine learning models + q-learning rl agent")
    run_modeling()

    step(5, "advanced analysis")
    try:
        run_advanced_analysis()
    except Exception as e:
        print(f"  advanced analysis had an issue: {e}")
        print("  you can run src/advanced_analysis.py separately if needed")

    total_time = time.time() - t_total
    print(f"  done | total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()