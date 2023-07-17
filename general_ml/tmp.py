import polars as pl

df = pl.DataFrame({"aaa": [1, 2, 3], "bbb": [5, 3, 6], "ccc": [5, 3, 6]})

df.select("*")[:2] = None


print(df.to_dict(False))