#  Personality Detection by Bidirectional LSTM
### Data:
- [Essays](http://web.archive.org/web/20160316113804/http://mypersonality.org/wiki/lib/exe/fetch.php?media=wiki:essays.zip)
- [GloVe Embeddings Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### Preprocessing:
- Dropped neutral samples
- Disclosure of reduction('s, 're, 'd, 've etc.)
- Filtered stop words
- Filtered outliers with a few words and too much words(100 <= number of words <= 500)

## Model
There is the model based on Bidirectional LSTM and GloVe Embeddings with output that contains 5 sigmoids.<br>
Models have greater accuracy **3-4%** on average then [related work based on CNN](https://sentic.net/deep-learning-based-personality-detection.pdf).

### Other models
Tried training more modern architecture such as BERT([Simple Transformers](https://github.com/ThilinaRajapakse/simpletransformers)) but model learned very simple rule that predict always 1.
