# Why is Winoground Hard? Investigating Failures in Visuolinguistic Compositionality
- Authors: [Anuj Diwan](https://ajd12342.github.io/), Layne Berry, [Eunsol Choi](https://www.cs.utexas.edu/~eunsol/), [David Harwath](https://www.cs.utexas.edu/~harwath/), [Kyle Mahowald](https://mahowak.github.io/)
- [EMNLP 2022 Paper](https://arxiv.org/abs/2211.00768)

:exclamation: This code release is still a work-in-progress: please raise an issue or send an email to `anuj.diwan@utexas.edu` and `layne.berry@utexas.edu` for questions.

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

## Section 3.1 Reproduction
First, run get_MODEL_scores.py for the model you're investigating to collect pairwise similarity scores for all 800x800 image-text combinations in Winoground.

Then, modify the variable "model" in recall_at_k.py to specify which model you are testing. Running recall_at_k.py will then output the R@1,2,5,10 scores for the I2T and T2I directions for the specified model.

## Section 3.2 Reproduction
First, run get_MODEL_feats.py for the model you're investigating to collect the embeddings at each layer for all 400x4 possible combinations of inputs within a Winoground set (I0+T0, I0+T1, I1+T0, and I1+T1).

Next, edit the "file_to_split" variable in split_train_test.py and run it to generate the stratified train and test splits we used.

Finally, run finetune_over_pooled_outputs/run_fused_embeddings_probe.py to perform the test. You can use the --train_path and --test_path command line parameters to specify the dataset to probe. A number of other command line parameters are available to configure the size of the probe, layer being probed, method for generating embeddings from a layer (i.e., CLS, Mean- or Max-Pooling), etc.

## Section 4 Reproduction
The Winoground sets assigned each newly introduced tag are provided in the file new_tag_assignments.json as a dictionary.

## Section 5.1 Reproduction
First, generate the augmented captions as specified above. Use get_MODEL_feats.py to generate embeddings of the augmentations.

Next, edit "file_to_split" in split_train_test.py and run it to generate the stratified train and test splits we used.

Finally, run SVC_linear_separability_by_example/probe_MODEL.py, where MODEL is UNITER, LXMERT, or CLIP. Use the command line arguments (visible via "--help" or at the top of the file) to configure the probe.

## Section 5.3 Reproduction
Generate augmented captions, embeddings of those captions, and the stratified train and test splits as for 5.1. Then, run full_dataset_separability/MODEL/run_unimodal_text_variants_probe.py to train and test a probe. Use the command line arguments (visible via "--help" or at the top of the file) to configure the probe.

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
Please also cite the wonderful paper that introduces the Winoground dataset:
```bibtex
@InProceedings{Thrush_2022_CVPR,
    author    = {Thrush, Tristan and Jiang, Ryan and Bartolo, Max and Singh, Amanpreet and Williams, Adina and Kiela, Douwe and Ross, Candace},
    title     = {Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5238-5248}
}
```
Feel free to contact `anuj.diwan@utexas.edu`  and `layne.berry@utexas.edu` with any questions!

