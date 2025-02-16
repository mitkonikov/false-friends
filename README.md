# Cross-Lingual False Friend Classification

Authors: Mitko Nikov, Žan Tomaž Šprajc, Žan Bedrač

[Link to the published paper on the SCORES 2024 Conference.](https://press.um.si/index.php/ump/catalog/book/886/chapter/147)

## Abstract

In this paper, we propose a novel approach to exploring cross-linguistic connections,
with a focus on false friends, using Large Language Model embeddings and graph databases.
We achieve a classification performance on the Spanish-Portuguese false friend dataset
of F1 = 83.81% using BERT and a multi-layer perceptron neural network.
Furthermore, using advanced translation models to match words between vocabularies,
we also construct a ground truth false friends dataset between Slovenian and Macedonian - two languages
with significant historical and cultural ties. Subsequently, we construct a graph-based representation
using a Neo4j database, wherein nodes correspond to words,
and various types of edges capture semantic relationships between them.

## To recreate the results

First change the directory to the `./slovenian-macedonian/src` directory. Afterwards you can run the following command:

```bash
python __main__.py classify --false_friends ./../false_friends.txt --true_friends ./../true_friends.txt
```

For more information on the command you can run:

```bash
python __main__.py classify -h
```

## References

Some utility functions for printing the measures were taken from [Santiago Castro, Jairo Bonanata and Aiala Rosá's repository](https://github.com/pln-fing-udelar/false-friends) to match their style of outputs.

## Citation

```bibtex
@book{Nikov2024Oct,
	author = {Nikov, Mitko and Šprajc, Žan Tomaž and Bedrač, Žan},
	title = {{Cross-Lingual False Friend Classification via LLM-based Vector Embedding Analysis}},
	year = {2024},
	month = oct,
	publisher = {Univerzitetna založba Univerze v Mariboru},
	doi = {10.18690/um.feri.6.2024}
}
```

> [!NOTE]
> The symbols `ČčŽžŠš` should be exchanged with proper LaTeX wrappers.
