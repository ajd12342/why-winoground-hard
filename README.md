# Why is Winoground Hard? Investigating Failures in Visuolinguistic Compositionality
- Authors: [Anuj Diwan](https://ajd12342.github.io/), Layne Berry, [Eunsol Choi](https://www.cs.utexas.edu/~eunsol/), [David Harwath](https://www.cs.utexas.edu/~harwath/), [Kyle Mahowald](https://mahowak.github.io/)
- [EMNLP 2022 Paper](https://arxiv.org/abs/2211.00768)

## Setup
```bash
conda create -n winoground python=3.6.9
conda activate winoground
pip install -r requirements.txt
```
Next, follow the installation instructions at `https://github.com/GEM-benchmark/NL-Augmenter` to install NL-Augmenter.

Download the Winoground dataset from `https://huggingface.co/datasets/facebook/winoground`. Place `examples.jsonl` and the extracted directory `images` inside `dataset`. 
## Generating Augmented Examples
Run
```bash
python augmentations/text_augmentations.py
```
to find the generated file at `examples_augmented.jsonl`.

## UNITER
For all experiments, when running UNITER code, first start a Docker container using the following code, then proceed as normal. All absolute paths `/path` are now `/slash/path`
```bash
bash UNITER/launch_container_simpler.sh
bash UNITER/run_init_docker.sh
```

## Citations and Contact
Please cite our paper if you use our paper, code, finegrained Winoground tags or the augmented Winoground examples in your work:
```bibtex
@inproceeding{diwan2022lwhywinogroundhard,
  author = {Diwan, Anuj and Berry, Layne and Choi, Eunsol and Harwath, David and Mahowald, Kyle},  
  title = {Why is Winoground Hard? Investigating Failures in Visuolinguistic Compositionality},
  year = 2022,
  booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
  publisher = "Association for Computational Linguistics",
}
```

