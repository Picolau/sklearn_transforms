from sklearn.base import BaseEstimator, TransformerMixin

# All sklearn Transforms must have the `transform` and `fit` methods
class Categorizador(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        
        def categorizar_horas(elem):
            if (elem >= 17 and elem <= 25):
                return 2
            elif (elem < 17 and elem >= 6):
                return 1
            else:
                return 0

        def categorizar_faltas(elem):
            if elem <= 3:
                return 2
            elif elem <= 6:
                return 1
            else:
                return 0

        data["H_AULA_PRES"] = data.iloc[:,9].apply(categorizar_horas)
        data["FALTAS"] = data.iloc[:,11].apply(categorizar_faltas)
        data["REPROVOU"] = data.iloc[:,0:4].apply(lambda row: bool(row[0] or row[1] or row[2] or row[3]), axis=1)
        
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
