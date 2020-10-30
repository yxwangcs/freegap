# Freegap

Install dependencies via `pip install -e .` and then issue `python -m freegap -h` to see the following help message:

```
usage: __main__.py [-h] [--datasets DATASETS] [--output OUTPUT] [--clear] [--counting] algorithm

positional arguments:
  algorithm            The algorithm to evaluate, options are `All, AdaptiveSparseVector, AdaptiveEstimates,
                       GapSparseVector, GapTopK`.

optional arguments:
  -h, --help           show this help message and exit
  --datasets DATASETS  The datasets folder
  --output OUTPUT      The output folder
  --clear              Clear the output folder
  --counting           Set the counting queries
```
