# Raskls
Richard's Auxiliary Sci-Kit Learn Stuff. Various homemade estimators I use often enough that I wanted to wrap them all up.

### Contents:
  - Tranformers
    - ColumnAutoTransformer
      - _Hands-off preprocessor. Interprets features and performs transformations according to their type (numerical, categorical, etc.), with sensible defaults and easy tweaking._
    - DictionaryTransformer (Comming Soon)
      - _Like builtin [sklearn.preprocessing.FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html), but for discrete dictionary mappings._
  - Optimizers
    - RandomSeedSearchCV
      - _Variation on RandomSearchCV. Enables easier searching across large or abnormal parameter spaces by calling a user-provided model-generating function to produce randomized models with only a random seed as input._
