import pandas as pd  # type: ignore
from pandas import DataFrame


def test_list() -> None:
    my_list = ["item1", "item2", "item3"]
    df = DataFrame(my_list, columns=["Column_Name"])
    print("\n", df)

    print("表1:")
    arr = [[1, 2, 3, 4, 5]]
    df = DataFrame(arr, columns=["a", "b", "c", "d", "e"])
    print(df)
    print(type(df.columns), df.columns)

    print("表2:")
    arr = [[11, 12, 13, 14, 15]]
    df = DataFrame(arr, dtype=float)
    print(df)
    print(type(df.columns), df.columns)


def test_cols() -> None:
    arr = [[1, 2, 3, 4, 5]]
    df1 = DataFrame(arr)  # , columns=['a', 'b', 'c', 'd', 'e'])
    print("")
    print(df1)
    print(df1.columns)
    df1.to_csv("t.csv", index=False)

    df2 = pd.read_csv("t.csv")
    print(df2)
    print(df2.columns)

    df3 = pd.concat([df1, df2], ignore_index=True)
    print(df3.columns)
    print(df3)


def test_append() -> None:
    df1 = DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df2 = DataFrame({"col1": [5, 6], "col2": [7, 8]})

    df3 = pd.concat([df1, df2])
    df4 = pd.concat([df1, df2], ignore_index=True)

    print(df1)
    print(df2)
    print(df3)
    print(df4)
