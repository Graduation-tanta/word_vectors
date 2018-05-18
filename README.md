# Word Vectors
My implementations for word vector representation algorithms (word2vec and glove) on text8 dataset 
I also implemented word2vec evaluation as in the paper but the results aren't good and needs more tuning that I leave as future work for me :(
I pushed my trained wordvectors in the same repo :3 only 15mb .. so small :)

## File Structure
    .
    ├── eval_data               
    │   ├── questions-phrases.txt           
    │   └──questions-words.txt         
    ├── glove                    # glove model
    |   ├── glove_temp           # my trained glove word vectors  
    │   └── myglove.py                # glove model
    ├── word2vec                    # Test files (alternatively `spec` or `tests`)
    │   ├── mygensim.py        # word2vec gensim model
    │   └── tf_myword2vec.py        # my word2vec model
    ├── tensorboard visualization.py # saves a checkpoint for tensorboard
    └── eval_wordvectors.py     # evaluation script for wordvectors and demo of KNN
    
## k nearest neighbours are still funny :D

```
one
=====================================
one                  1.0000
eight                0.9855
seven                0.9838
six                  0.9809
nine                 0.9799
four                 0.9773
five                 0.9769
three                0.9717
two                  0.9651
zero                 0.9605

fish
=====================================
fish                 1.0000
aquarium             0.3756
farming              0.3731
populations          0.3715
products             0.3617
meat                 0.3462
food                 0.3401
remote               0.3191
eating               0.3190
animals              0.3181

cairo
=====================================
cairo                1.0000
strikers             0.3217
egypt                0.3199
algeria              0.3044
muhammed             0.2954
iran                 0.2923
zoo                  0.2831
amaranth             0.2795
orleans              0.2765
kantele              0.2755

human
=====================================
human                1.0000
nature               0.6536
rights               0.6256
behavior             0.6133
animal               0.5955
or                   0.5950
individual           0.5948
particular           0.5872
any                  0.5736
such                 0.5730

mohammed
=====================================
mohammed             1.0000
dora                 0.3219
shah                 0.3163
frederic             0.3102
alexius              0.3079
vin                  0.3073
irradiation          0.3006
tires                0.2967
westcott             0.2941
godard               0.2902

time
=====================================
time                 1.0000
when                 0.8355
only                 0.8259
this                 0.8234
but                  0.8196
since                0.8124
while                0.8050
it                   0.8031
however              0.8028
which                0.8006

standards
=====================================
standards            1.0000
organizations        0.4634
international        0.4552
standard             0.4307
uses                 0.4016
types                0.3887
methods              0.3741
industry             0.3650
internet             0.3638
common               0.3585

program
=====================================
program              1.0000
programs             0.6051
computer             0.5709
system               0.5676
for                  0.5636
based                0.5591
provides             0.5435
software             0.5316
information          0.5182
using                0.5115

machine
=====================================
machine              1.0000
machines             0.5007
gun                  0.4386
tools                0.4355
translation          0.4120
computer             0.4078
using                0.3988
guns                 0.3985
purpose              0.3958
method               0.3944

intelligence
=====================================
intelligence         1.0000
artificial           0.5564
agency               0.4820
service              0.4342
security             0.4068
military             0.3907
secret               0.3791
department           0.3684
agents               0.3684
national             0.3607
```

## Resources

### glove
* [glove paper](https://nlp.stanford.edu/pubs/glove.pdf)
* [blog post](http://www.foldl.me/2014/glove-python/)
* [nice tensorflow implementation I looked at](https://github.com/GradySimon/tensorflow-glove)

### word2vec
* [http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/]
* [http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/]
* [https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb]
* [http://distill.pub/2016/misread-tsne/]


