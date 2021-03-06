# Nessy: a Neuro-Symbolic System for Label Noise Reduction

- **Overview**: Nessy is a new neuro-symbolic system that takes advantage of *deep* and *probabilistic* modelling for inferring  latent classes, 
and of *symbolic knowledge* expressed as a set of logic rules for improving model inference. Unlike previous probabilistic methods, our deep probabilistic model 
adopts deep neural networks to parameterize data distributions, thus is able to model complex feature relationships in the data and to learn high-quality 
latent feature representations.

- **Datasets**: For our experiments, we use [TAC Relation Extraction Dataset](https://nlp.stanford.edu/projects/tacred/) available for download from [LDC TACRED webpage](https://catalog.ldc.upenn.edu/LDC2018T24).


### Prerequisites

#### Data

TACRED can be downloaded [here](https://catalog.ldc.upenn.edu/LDC2018T24).

Supplemental data can be downloaded [here](https://drive.google.com/drive/folders/1YVL-T_UAxTNLg6t9OU71c2I33yH825Tc?usp=sharing).
It includes the data for all 4 datasets that we experimented with:

  - title_random: Title dataset with random noise (noise ratio is 40%);
  - title_ds: Title dataset with distant supervision noise;
  - empoyee_ds: Employee dataset with distant supervision noise;
  - top_members_ds: Top Members dataset with distant supervision noise.

For each dataset, the following files are provided:

  - vocabulary
  - pre-trained embeddings for train, development and test data. The embeddings were obtained using [PALSTM model](https://github.com/yuhaozhang/tacred-relation)
  - linking results: output of the entity linking tool (we used [BLINK](https://github.com/facebookresearch/BLINK))

To sample the dataset as mentioned is the paper run the following commands:

```
python instance_selection.py /path/to/tacred/train.json per:title data/title/train.json --head_type=PERSON --tail_type=TITLE
python instance_selection.py /path/to/tacred/dev.json per:title data/title/dev.json --head_type=PERSON --tail_type=TITLE
python instance_selection.py /path/to/tacred/test.json per:title data/title/test.json --head_type=PERSON --tail_type=TITLE
```

To add random noise run the following command:

```
python add_random_noise.py data/title/train.json data/title_40/train.json per:title --percent 0.4 --seed 31
```

#### Packages


### Run pre-training


### Run training


### Run evaluation
