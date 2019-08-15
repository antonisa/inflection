# A Tool for Morphological Inflection

This is the code accompanying our paper on "Pushing the Limits of Low-Resource Morphological Inflection".

# Requirements

Are listed in `requirements.txt` so you can just run
~~~
pip install -r requirements.txt
~~~

# Hallucinating data

Use the `augment.py` script as follows:
~~~
python augment.py L1
~~~

# Attribution

The `align.py`, `align.c` and `Makefile` are taken from Roee Aharoni's work: https://github.com/roeeaharoni/morphological-reinflection/tree/master/src


# Citation
If you use this tool for your work, please consider citing the corresponding paper "Pushing the Limits of Low-Resource Morphological Inflection", Antonios Anastasopoulos and Graham Neubig, to appear at EMNLP 2019.

bibtex:
~~~
@inproceedings{anastasopoulos19emnlp,
    title = {Pushing the Limits of Low-Resource Morphological Inflection},
    author = {Anastasopoulos, Antonios and Neubig, Graham},
    booktitle = {Proc. EMNLP},
    address = {Hong Kong},
    month = {November},
    year = {2019}
}
~~~

