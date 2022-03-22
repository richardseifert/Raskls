import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer

class ColumnAutoTransformer(BaseEstimator):
    '''
    Hands-off data preprocessing tools for dynamically classifying
    columnar data and transforming numerical, categorical,
    boolean, and textual data features with separate, refineable,
    preprocessing subroutines.
    Responsible for:
      - Categorizing each feature as one of the following kinds:
          - Numerical (int or float, many unique values)
          - Categorical (string int or float, few unique values)
          - Boolean (string int or float, two unique values)
          - text (string, many unique values)
      - Imputing, scaling, and encoding features according to their kind.
          - Sensible defaults
             - Median imputing for numerical features, otherwise most frequent.
             - Standard scaler for numerical features, otherwise none.
             - OneHot encoding for categorical features, Ordinal for boolean features.
             - Tfidf text encoding.
          - User refinements
             - imputers, scalers, and encoders can be manually set for entire feature
               categories (numerical, categorical, etc.), or for specific features.
    '''
    instantiatable_keywords = {'Simple':SimpleImputer,
                               'Standard':StandardScaler,
                               'OneHot':OneHotEncoder,
                               'Ordinal':OrdinalEncoder,
                               'Count':CountVectorizer,
                               'Tfidf':TfidfVectorizer,
                               'Hashing':HashingVectorizer}

    def __init__(self,override=None,categorical_thresh=10,
                   imputer="Simple",impute_passthroughs=False,
                   impute_numerical="median",impute_categorical="most_frequent",
                   impute_boolean="most_frequent",impute_textual="most_frequent",
                   scaler="Standard", scale_encoded_text=False,
                   cat_encoder="OneHot",bool_encoder="Ordinal",
                   txt_encoder="Tfidf",txt_n_features=20,
                   drop=None,passthrough=None,verbose=False):
        #Set feature classification parameters
        self.override = override
        self.categorical_thresh = categorical_thresh

        #Set imputing parameters
        self.imputer = imputer
        self.impute_numerical = impute_numerical
        self.impute_categorical = impute_categorical
        self.impute_boolean = impute_boolean
        self.impute_textual = impute_textual
        self.impute_passthroughs=impute_passthroughs

        #Set scaling parameters
        self.scaler = scaler
        self.scale_encoded_text = scale_encoded_text

        #Set encoding parameters
        self.cat_encoder= cat_encoder
        self.txt_encoder= txt_encoder
        self.txt_n_features=txt_n_features
        self.bool_encoder=bool_encoder

        #Set ignore parameters
        self.drop = drop
        self.passthrough = passthrough

        #Set verbosity!
        self.verbose = verbose #TODO: add verbosity throughout.

    class _Decorators:
        def get_transformer(func):
            def wrapper(self, feat, kind=None):
                if kind is None:
                    kind = self._feature_kind(feat)
                tfm = func(self, feat, kind)
                return self._instantiate_from_keyword(tfm)
            return wrapper

    def set_params(self, *args, **kwargs):
        # Do this by re-calling __init__ (hopefully kosher).
        # See __init__ for parameter info.
        self.__init__(*args,**kwargs)

    def _feature_kind(self, feat):
        is_overridden,override_value = self._get_override(feat.name, 'kind')
        if is_overridden:
            return override_value

        #Identify number of unique values
        N_unique = len(np.unique(feat[~pd.isna(feat)]))

        #Identify feature type!
        try:
            float(feat[~pd.isna(feat)].iloc[0])
            kind = 'numerical'
        except ValueError:
            kind = 'textual'
        if N_unique == 2:
            kind = 'boolean'
        elif N_unique <= self.categorical_thresh:
            kind = 'categorical'
        return kind

    def _instantiate_from_keyword(self, keyword, **kwargs):
        '''
        Helper for grabbing sub-estimator instances.
        Tranforms keyword into estimator by the following cases:
          1. If known keyword, return new estimator instance for that keyword
          2. If unknown keyword, raise error.
          3. If estimator, pass estimator.
          4. If None, pass None.
        ARGUMENTS:
          keyword  - string, object, or None.
          **kwargs - Arguments to instantiate with, for string keywords.
        '''
        if isinstance(keyword,str):
            if keyword in self.instantiatable_keywords:
                return self.instantiatable_keywords[keyword](**kwargs)
            raise ValueError("Unknown keyword: %s"%(keyword))
        return keyword

    def _get_override(self, feat, overrideable):
        '''
        Get override info for a specific feature and overridable thing.
        Returns overridden status and override value (None if not overridden).
        ARGUMENTS:
          feat - String feature name.
          overrideable - String identifier of the thing to check override status.
                        Supported: 'kind','imputer','scaler','encoder'
        '''
        try:
            return True,self.override[feat][overridable]
        except (TypeError,KeyError) as e:
            return False,None

    def impute_strategy(self, kind):
        if kind == 'numerical':
            return self.impute_numerical
        if kind == 'categorical':
            return self.impute_categorical
        if kind == 'boolean':
            return self.impute_boolean
        if kind == 'textual':
            return self.impute_textual
        raise ValueError("Unrecognized impute kind, %s"%(kind))

    @_Decorators.get_transformer
    def _get_imputer(self, feat, kind=None):
        is_overridden,override_value = self._get_override(feat.name,'imputer')
        if not is_overridden:
            if self.imputer == 'Simple':
                return self._instantiate_from_keyword(self.imputer,
                               strategy=self.impute_strategy(kind))
            else:
                return self._instantiate_from_keyword(self.imputer)
        else:
            return override_value
    @_Decorators.get_transformer
    def _get_scaler(self, feat, kind=None):
        is_overridden, override_value = self._get_override(feat.name,'scaler')
        if not is_overridden:
            if kind == 'numerical' or (kind == 'textual' and self.scale_encoded_text):
                return self.scaler
            return None
        else:
            return override_value

    @_Decorators.get_transformer
    def _get_encoder(self, feat, kind=None):
        is_overridden,override_value = self._get_override(feat.name,'encoder')
        if not is_overridden:
            if kind == 'categorical':
                return self.cat_encoder
            elif kind == 'textual':
                return self.txt_encoder
            elif kind == 'boolean':
                return self.bool_encoder
            else:
                return None
        else:
            return override_value

    def _make_info(self, X, store=True):
        '''
        Helper method to gather information about each feature in dataset X.
        Info table also houses sub-estimator instances used in fit, transform.
        ARGUMENTS:
          X     - 2D array-like. Dataset to construct info table about.
          store - Flag to store output in model variable column_info_, which
                  is used in the transform method.
        '''
        X = self._validate_X(X)
        passthrough = self.passthrough if (not self.passthrough is None) else []
        drop = self.drop if (not self.drop is None) else []
        df = pd.DataFrame(columns=['ID','dtype','kind','imputer','scaler','encoder'])
        for feat in X.columns:
            entry = {}
            entry['ID'] = feat
            entry['dtype'] = X[feat].dtype
            kind = self._feature_kind(X[feat]) if (not feat in drop) else 'drop'
            entry['kind'] = kind
            if feat in passthrough or feat in drop:
                entry['imputer'] = None
                entry['scaler'] = None
                entry['encoder'] = None
            else:
                entry['imputer'] = self._get_imputer(X[feat],kind)
                entry['scaler'] = self._get_scaler(X[feat],kind)
                entry['encoder'] = self._get_encoder(X[feat],kind)
            df = df.append(entry,ignore_index=True)
        if store:
            self.column_info_ = df
        return df

    def info(self, X=None):
        '''
        User-facing info gathering method. Returns feature information
        known by the fitted model.
        Can also be hijacked to generate info for a user-provided dataset, X.
        Info table includes the following:
          ID      : Feature name or column index.
          dtype   : Data type of feature.
          kind    : Feature type. Used to assign sub-estimators.
                    One of 'numerical','categorical','boolean','textual','drop'.
          imputer : Imputer instance for this feature, or None.
          scaler  : Scaler instance for this feature, or None.
          encoder : Encoder instance for this feature, or None.
        ARGUMENTS:
          X - [Optional] 2D array-like. Data to gather info on.
              If nothing provided, table will reflect info on
               the training data used to fit this model.
        '''
        if X is None:
            check_is_fitted(self)
            return self.column_info_
        else:
            return self._make_info(X, store=False)

    def _validate_X(self, X):
        '''
        Check that X has proper shape. Cast to pandas if not already.
        '''
        #TODO: Check shape.
        if not isinstance(X,pd.DataFrame):
            return pd.DataFrame(X)
        return X

    def _validate_output_format(self, output_format):
        #Check that user-requested output format is supported.
        if output_format in ['pandas','numpy']:
            return output_format
        else:
            raise ValueError("Invalid output_format '%s'. Must be on of: 'pandas', 'numpy'"%(output_format))

    def _cast_label_transformed(self, xt, label, sub_labels=None):
        '''
        Cast a transformed feature to pandas DataFrame, and label column(s).
        TODO: Use sub_labels instead of numbers for, e.g., OneHotEncoded columns.
              Not sure how to standardize this, or ensure that labels will be in
              the right order. Try l8rrrr
        '''
        try:
            xt = pd.DataFrame.sparse.from_spmatrix(xt)
        except AttributeError:
            xt = pd.DataFrame(xt)

        if sub_labels is None:
            sub_labels = [str(i) for i in range(len(xt.columns))]

        label_cols = xt.shape[1]>1
        for sl, col in zip(sub_labels, xt.columns):
            new = "%s%s"%(label, "-"+sl if label_cols else "")
            xt = xt.rename(columns={col: new})
        return xt

    def _format_output(self, Xt):
        '''
        TODO: based on self.output_format_, cast Xt to numpy or leave alone.
        For now, just return as Pandas.
        '''
        if isinstance(Xt,pd.DataFrame) and self.output_format_ == 'pandas':
            return Xt #Already correctly formatted as pandas
        elif self.output_format_ == 'pandas':
            return pd.DataFrame(Xt) #Needs reformatting to pandas
        elif isinstance(Xt,pd.DataFrame):
            return np.array(Xt) #Needs reformatting to numpy
        else:
            #Should be correctly formatted as numpy. But check.
            try:
                assert isinstance(Xt,np.ndarray)
            except AssertionError:
                raise TypeError("Unrecognized datatype for Xt. Should be pandas.DataFrame or numpy.ndarray.")
            return Xt

    def fit(self, X, y=None, output_format=None):
        '''
        Generate/retrieve sub-estimators, fit each with training data, and determine output format.
        ARGUMENTS:
          X - 2D array-like. Training data
          y - [Optional] Target variable. Not used here, but accepted for potential
                         model chaining.
          output_format - [Optional] Desired format for transformed X. One of 'pandas','numpy'.
                            By default, will match format of X.
        '''
        #Validate X
        X=self._validate_X(X)

        #Gather column info and instantiate subtransformers.
        self._make_info(X) #makes self.column_info_

        # fit all subtransformers.
        for i,feature_info in self.column_info_.iterrows():
            if feature_info['kind'] == 'drop':
                continue
            feat = feature_info['ID']
            #Transform feature!
            xt = pd.DataFrame(X[feat]).copy()
            if not feature_info['imputer'] is None:
                xt=feature_info['imputer'].fit_transform(xt)
            if not feature_info['scaler'] is None:
                xt=feature_info['scaler'].fit_transform(xt)
            try:
                if not feature_info['encoder'] is None:
                    xt=feature_info['encoder'].fit_transform(xt)
            except AttributeError:
                if not feature_info['encoder'] is None:
                    xt=feature_info['encoder'].fit_transform(xt.flatten())

        # Determine output format.
        if isinstance(X,pd.DataFrame):
            self.output_format_ = 'pandas'
        else:
            self.output_format_ = 'numpy'
        #Override if output_format provided.
        if not output_format is None:
            self.output_format_ = self._validate_output_format(output_format)
        return self

    def transform(self,X):
        '''
        Transform each feature using its corresponding sub-estimators.
        Return transformed data, Xt.
        ARGUMENTS:
          X - 2D array-like. Data to transform!
        '''
        check_is_fitted(self)

        #Validate X
        X=self._validate_X(X)

        xts = [] #to store transformed features.
        for i,feature_info in self.column_info_.iterrows():
            if feature_info['kind'] == 'drop':
                continue
            feat = feature_info['ID']
            #Transform feature!
            xt = pd.DataFrame(X[feat]).copy()
            if not feature_info['imputer'] is None:
                xt=feature_info['imputer'].transform(xt)
            if not feature_info['scaler'] is None:
                xt=feature_info['scaler'].transform(xt)
            try:
                if not feature_info['encoder'] is None:
                    xt=feature_info['encoder'].transform(xt)
            except AttributeError:
                #Text encoders need 1D input (for some reason... :( )
                if not feature_info['encoder'] is None:
                    xt=feature_info['encoder'].transform(xt.flatten())

            feat_names = None
            if isinstance(feature_info['encoder'], OneHotEncoder):
                feat_names = feature_info['encoder'].categories_[0]

            xts.append(self._cast_label_transformed(xt,str(feat), feat_names))
        Xt = pd.concat(xts,axis=1)
        return self._format_output(Xt)

    def fit_transform(self,X,y=None,**kwargs):
        '''
        Chained fit and then transform.
        '''
        return self.fit(X,y,**kwargs).transform(X,**kwargs)