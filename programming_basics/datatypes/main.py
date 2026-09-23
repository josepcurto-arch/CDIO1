import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np


def numpy_demo():
    foo = np.random.randn(10, 10)
    shape = foo.shape
    element_types = type(foo[0][0])
    print(f"This generates a {shape[0]}x{shape[1]} matrix of random {element_types.__name__} elements.")
    print(foo)


def matplotlib_demo():
    foo = np.random.randn(10, 10, 3)
    plt.figure()
    plt.title("10x10 Random RGB Image")
    plt.imshow(foo, interpolation='none')
    plt.show()

    plt.figure()
    plt.title("First row in the random matrix")
    plt.plot(foo[:, 0, 0], 'o')
    plt.show()


def pandas_demo():
    csv_lines = [
        "id,name,age,city",
        "1,Alice,30,New York",
        "2,Bob,25,Los Angeles",
        "3,Charlie,35,Chicago",
        "4,Diana,28,Houston"
    ]

    data = [line.split(",") for line in csv_lines]

    header = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=header)
    filtered_df = df[df['age'].astype(int) > 30]
    print(f"DataFrame with {len(df)} rows and {len(df.columns)} columns:")
    print(df)
    print("Filtered DataFrame (age > 30):")
    print(filtered_df)


def xarray_demo():
    data = xr.DataArray(
        np.random.rand(3, 4),
        dims=["time", "location"],
        coords={
            "time": pd.date_range("2025-06-01", periods=3),
            "location": ["A", "B", "C", "D"]
        },
        name="temperature"
    )

    print(data)

def pickle_demo():
    import pickle
    data = {'name': 'Alice', 'age': 30, 'city': 'New York'}
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open('data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    print("Pickled data:", loaded_data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scientific Libraries Demo")
    parser.add_argument("--numpy", action="store_true", help="Run NumPy demo")
    parser.add_argument("--matplotlib", action="store_true", help="Run Matplotlib demo")
    parser.add_argument("--pandas", action="store_true", help="Run Pandas demo")
    parser.add_argument("--xarray", action="store_true", help="Run xarray demo")
    parser.add_argument("--pickle", action="store_true", help="Run pickle demo")
    args = parser.parse_args()
    if args.numpy:
        numpy_demo()
    if args.matplotlib:
        matplotlib_demo()
    if args.pandas:
        pandas_demo()
    if args.xarray:
        xarray_demo()
    if args.pickle:
        pickle_demo()
    if not (args.numpy or args.matplotlib or args.pandas or args.xarray or args.pickle):
        print("Please specify at least one library to demo: --numpy, --matplotlib, --pandas, --xarray")
