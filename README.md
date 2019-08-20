# A Tool for Morphological Inflection

This is the code accompanying our paper on ["Pushing the Limits of Low-Resource Morphological Inflection".](https://arxiv.org/pdf/1908.05838.pdf)

# Requirements

Are listed in `requirements.txt` so you can just run
~~~
pip install -r requirements.txt
~~~

Also, run `make` in order to build the executable needed for data hallucination.

# Data Hallucination

Use the `augment.py` script as follows:
~~~
python augment.py [data_directory] [language] --examples N [--use_dev]
~~~

The script assumes data in the format of the [SIGMORPHON 2019 task 1 inflection shared task](https://sigmorphon.github.io/sharedtasks/2019/task1/) (example from Asturian):
~~~
meyorar	meyoraría	V;1;SG;COND
firir	firir	V;NFIN
algamar	algamareis	V;2;PL;SBJV;PST;IPFV;LGSPEC1
...
~~~

All scripts assume that files named `language-train`, `language-dev`, and `language-test` are under `data_directory`.
The output is a file `language-hall` under `data_directory` with `N` hallucinated examples.

If you want to also use the dev dataset for hallucination (recommended for extremely low-resource cases) add the `--use_dev` flag.

### Attribution
The `align.py`, `align.c` and `Makefile` are taken from Roee Aharoni's work: https://github.com/roeeaharoni/morphological-reinflection/tree/master/src

# Training inflection models

The main script is `inflection.py` which implements the models and handles training, testing, etc.
For standard training using cross-lingual transfer, run:
~~~
py inflection.py \
	--datapath sample-data/ \
	--L1 adyghe \
	--L2 kabardian \
	--mode train \
	--setting original
~~~

Running the above command trains for about 40 minutes on a single CPU (2.4 GHz), producing the following output

~~~
[dynet] random seed: 2846648232
[dynet] allocating memory: 512MB
[dynet] memory allocation done.
Data lengths
transfer-language 10000 10000 10000
...
...
Accuracy good enough, breaking
[lr=0.1 clips=3401 updates=14115] Epoch  0  :  28131.39509539181
	 COPY Accuracy:  0.95  average edit distance:  0.05
	 TASK Accuracy:  0.68  average edit distance:  0.6
...
...
Epoch  14  :  0.5778098778155254
[lr=0.0125 clips=1 updates=100] 	 COPY Accuracy:  1.0  average edit distance:  0.0
	 TASK Accuracy:  0.7  average edit distance:  0.58
Restarting the trainer with half the learning rate!
Best dev accuracy after finetuning:  0.7
Best dev lev distance after finetuning:  0.58
~~~
The above script should produce three models under `./models` (based on the dev performance metrics).

After training, you can use these models to produce output on the test files, as follows:
~~~
py inflection.py \
	--datapath sample-data/ \
	--L1 adyghe \
	--L2 kabardian \
	--mode test \
	--setting original
~~~

## Additional Notes

Using various flags you can:

* use multiple L1 transfer languages by separating the languages with commas e.g. `--L1 adyghe,armenian`.

* specify the location to store the models by using the `--modelpath` flag

* specify the location to store the output (when testing) by using the `--outputpath` flag

* tell the model to use hallucinated data (as created by the `augment.py` script above) with the `--use_hall` flag.

* train a model using only hallucinated data with the `--only_hall` flag.

* train a model using only the low-resource language (hence without cross-lingual transfer) by setting `--setting low`

* swap the low-resource data, using the dev set for training, and using the train as a dev set, by setting `--setting swap`

* toggle the language discriminator, and lemma and tag attention regularization components on (they are disabled by default) by using `--predict_lang`, `--use_att_reg`, and `--use_tag_att_reg`. All of them are recommended for best performance.

* get outputs using various ensemble combinations by using `--mode test-ensemble` (uses the three models produced by a single run) or `--mode test-all-enssemble` (uses four models, two produced by a `--setting original` run and two by a `--setting swap` run).

* produce attention visualizations over the development set by setting `--mode draw-dev` and providing a path to store the figures through `--figurepath`. Requires `matplotlib`.


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
    year = {2019},
    note = {to appear}
}
~~~

