#  Personality Detection by LSTM
### Data:
- [Essays](http://web.archive.org/web/20160316113804/http://mypersonality.org/wiki/lib/exe/fetch.php?media=wiki:essays.zip)
- [GloVe Embeddings Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### Preprocessing:
- Dropped neutral samples
- Disclosure of reduction('s, 're, 'd, 've etc.)
- Filtered stop words
- Filtered outliers with a few words and too much words(100 <= number of words <= 500)

## Model
There are five models based on Biderectional LSTM and GloVe Embeddings.<br>
Models have greater accuracy **3-4%** on average then [related work based on CNN](https://sentic.net/deep-learning-based-personality-detection.pdf).
