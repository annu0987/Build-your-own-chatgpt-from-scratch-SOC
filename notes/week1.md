# Week 1 Notes - Python and DS Tools

## What this week was about

The first week was mostly setup. Python refresher, numpy, pandas, matplotlib. I already knew basic Python so the refresher video was mostly background noise, but I watched it at 1.5x just to make sure I wasn't missing anything.

---

## NumPy

The main thing I took away is that numpy lets you do math on entire arrays without writing loops. Like if you have an array of numbers and you want to multiply each one by 2, you just write `arr * 2` and it works. This is called vectorization I think.

Broadcasting was the confusing part. When you add a (3, 1) array and a (1, 4) array, numpy figures out how to stretch them to match. It works but I had to read about it twice before it clicked.

Useful things I noted:
- `np.zeros`, `np.ones`, `np.random.randn` for creating arrays
- Slicing works like Python lists but in multiple dimensions
- `arr.shape` and `arr.reshape()` come up constantly

---

## Pandas

Pandas is basically spreadsheets in Python. A DataFrame is rows and columns, similar to Excel. I mostly used it to load CSVs and check what the data looks like.

`df.head()`, `df.info()`, `df.describe()` - these three together tell you a lot about a dataset quickly.

---

## Matplotlib

Didn't spend too long here. Enough to plot a line chart and a histogram. The syntax is a bit weird (`plt.subplot` and figuring out axes) but once you get the pattern it's fine.

---

## Personal observations

One thing I noticed is that a lot of the "real" ML work is just moving data around in the right shape. Like half the errors I got while experimenting were shape mismatches between arrays. So numpy basics actually matter more than they look like they do.

Also I realized I didn't really know what a matrix multiplication was doing geometrically until I looked it up separately. The videos don't explain that but it helped to think of it as a transformation.
