######
submit dev f1:
processed 16261 tokens with 1123 phrases; found: 822 phrases; correct: 404.
accuracy:  33.60%; (non-O)
accuracy:  94.07%; precision:  49.15%; recall:  35.98%; FB1:  41.54
          company: precision:  41.46%; recall:  34.69%; FB1:  37.78  41
         facility: precision:  54.55%; recall:  38.96%; FB1:  45.45  55
          geo-loc: precision:  53.29%; recall:  56.33%; FB1:  54.77  167
            movie: precision:  23.08%; recall:  10.00%; FB1:  13.95  13
      musicartist: precision:  25.00%; recall:   5.33%; FB1:   8.79  16
            other: precision:  25.96%; recall:  23.58%; FB1:  24.71  208
           person: precision:  67.87%; recall:  64.26%; FB1:  66.02  249
          product: precision:  45.24%; recall:  12.03%; FB1:  19.00  42
       sportsteam: precision:  62.07%; recall:  21.95%; FB1:  32.43  29
           tvshow: precision:  50.00%; recall:  50.00%; FB1:  50.00  2
######


Note that you are advised to use RNN, GRU or LSTM models for this assignment, and Word Embedding should be used.
The evaluation scores of submission are not the only metrics.
Code integrity will also be considered a critical part.
Please do not make large copies of code snippets from GitHub or various tutorials.
 
You can use 
https://github.com/sighsmile/conlleval (python) or http://www.cnts.ua.ac.be/conll2000/chunking/conlleval.txt (Perl)
to evaluate your model's performance on dev / evaluating data.
 
If you are not satisfied with the model performance,you can use external data (e.g., a large amount of twitter data) to train word embeddings or directly use word embeddings such as Glove and Word2Vec pretrained by other researchers.


"train" and "dev" are annotated with 10 fine-grained NER categories: person, geo-location, company, facility, product, music artist, movie, sports team, tv show and other.  

Baseline(CRF) results for reference:

10 Entity Types:
accuracy:  93.68%; precision:  40.34%; recall:  32.22%; F1:  35.83
          company: precision:  43.48%; recall:  25.64%; F1:  32.26
         facility: precision:  19.44%; recall:  18.42%; F1:  18.92
          geo-loc: precision:  49.18%; recall:  51.72%; F1:  50.42
            movie: precision:  16.67%; recall:   6.67%; F1:   9.52
      musicartist: precision:   0.00%; recall:   0.00%; F1:   0.00
            other: precision:  28.57%; recall:  18.18%; F1:  22.22
           person: precision:  52.04%; recall:  59.65%; F1:  55.59
          product: precision:  12.00%; recall:   8.11%; F1:   9.68
       sportsteam: precision:  33.33%; recall:   8.57%; F1:  13.64
           tvshow: precision:   0.00%; recall:   0.00%; F1:   0.00

Note that this is just an example output for evaluating metrics. In this assignment, it is normal for the F1-score to be around 20~30.
