import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


from scipy.stats import entropy
from gtda.time_series import TakensEmbedding, SingleTakensEmbedding
from sklearn.decomposition import PCA
import mtd 
from scipy.stats import entropy

from tqdm import tqdm

def my_entropy(cross_barcodes, normalize = False):
    pers_entropy = np.zeros((len(cross_barcodes),2))

    sum_lifespan = np.zeros(len(cross_barcodes))

    
    for barcode_idx in range(len(cross_barcodes)):
        for hom_dim in [0, 1]:
            lifespan_sums = cross_barcodes[barcode_idx][hom_dim][:, 1] - cross_barcodes[barcode_idx][hom_dim][:, 0]
            sum_lifespan[barcode_idx] += np.sum(lifespan_sums)

            if len(lifespan_sums) != 0:
                entropy_dim = entropy(lifespan_sums.astype(float), base = 2)
            else:
                entropy_dim = 0
            
            pers_entropy[barcode_idx][hom_dim] = entropy_dim

    if normalize:
        pers_entropy /= np.log2(sum_lifespan[...,None])
        
    return pers_entropy



class TopologicalFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 representatives,
                 embedding_dimension = 50,
                 embedding_time_delay = 4,
                 search_opt_embd = True,
                 stride = 5,
                 n_components = 3,
                 random_state = 42,
                 verbose = True):
        
        assert len(representatives) != 0, "Oh no! This assertion failed!"
        assert representatives[0].shape[-1] > (embedding_dimension - 1) * embedding_time_delay, "Oh no! You need lower dim!!!"
        
        self.search_opt_embd = search_opt_embd
        if search_opt_embd:
            embedder = SingleTakensEmbedding(
                parameters_type="search",
                n_jobs=-1,
                stride=stride,
                time_delay=embedding_time_delay,
                dimension=embedding_dimension,
                    )
            embedder.fit(representatives[0].squeeze())
            self.embedding_dimension = embedder.dimension_
            self.embedding_time_delay = embedder.time_delay_
            
            if verbose:
                print('Optimal time delay based on mutual information:', embedder.time_delay_)
            if verbose:
                print('Optimal embedding dimension based on false nearest neighbors:', embedder.dimension_)
        else:
            self.embedding_dimension = embedding_dimension
            self.embedding_time_delay = embedding_time_delay

        self.stride = stride
        self.n_components = n_components if self.embedding_dimension > n_components else self.embedding_dimension

        
        self.embedder = TakensEmbedding(time_delay=embedding_time_delay,
                           dimension=embedding_dimension,
                           stride=stride)
        
        self.PCA = PCA(n_components=self.n_components)
        self.persistence = mtd.calc_cross_barcodes 
        self.entropy = my_entropy
        self.representatives = representatives
        self.random_state = random_state
        self.verbose = verbose
        np.random.seed(self.random_state)

        self.representatives_transformed = np.array(self.embedder.fit_transform(self.representatives))

        self.representatives_transformed= [self.PCA.fit_transform(cloud) for cloud in self.representatives_transformed]
        
        if verbose:
            print(f"The result shape of cloud: {self.representatives_transformed[0].shape}")

    def fit(self, X, y, batch_size_L = 6000, batch_size_R = 6000):
        return self

    def transform(self, X, batch_size_L = 6000, batch_size_R = 6000):

        all_features = len(self.representatives)*8 + 4
        ent_features = len(self.representatives)*4
        
        features = np.array(self.embedder.fit_transform(X))

        # Maybe we need to add somme kind of normalization here

        features = [self.PCA.fit_transform(cloud) for cloud in features]
        # Is it optimal to use PCA here?

        top_features = np.empty([len(features), all_features])
        top_features_names = []


        i = 0
        j = 0

        with tqdm(total=len(self.representatives_transformed)*2 + 1, disable = not self.verbose) as pbar:
            for representative in self.representatives_transformed:
                cross_barcodes = []
    
                for cloud in features:
                    cross_barcodes.append(self.persistence(representative, cloud,
                                                      batch_size1 = batch_size_L,
                                                      batch_size2 = batch_size_R,
                                                      pdist_device = "cuda", is_plot = False))
                
                entropyes = self.entropy(cross_barcodes, normalize=False)
                
                mtd_features = [mtd.get_score(barc, 1, 'sum_length') for barc in cross_barcodes]
                mtd_features = np.array(mtd_features).reshape(-1,1)
                top_features[:, ent_features+j] = mtd_features.copy().squeeze()

                mtd_features = [mtd.get_score(barc, 0, 'sum_length') for barc in cross_barcodes]
                mtd_features = np.array(mtd_features).reshape(-1,1)
                top_features[:, ent_features+j+1] = mtd_features.copy().squeeze()
                
                top_features[:, i:i+2] = entropyes.copy()
                # top_features[:, 8+j] = mtd_features.copy().squeeze()
                
                pbar.update(1)
                i += 2
                j += 2
                
                cross_barcodes = []
    
                for cloud in features:
                    cross_barcodes.append(self.persistence(cloud, representative, 
                                                      batch_size1 = batch_size_L,
                                                      batch_size2 = batch_size_R,
                                                      pdist_device = "cuda", is_plot = False))
                
                entropyes = self.entropy(cross_barcodes, normalize=False)
                
                mtd_features = [mtd.get_score(barc, 1, 'sum_length') for barc in cross_barcodes]
                mtd_features = np.array(mtd_features).reshape(-1,1)
                top_features[:, ent_features+j] = mtd_features.copy().squeeze()

                mtd_features = [mtd.get_score(barc, 0, 'sum_length') for barc in cross_barcodes]
                mtd_features = np.array(mtd_features).reshape(-1,1)
                top_features[:, ent_features+j+1] = mtd_features.copy().squeeze()
                
                top_features[:, i:i+2] = entropyes.copy()
                # top_features[:, 8+j] = mtd_features.copy().squeeze()
                
                pbar.update(1)
                i += 2
                j += 2

            barcodes = []
            for cloud in features:
                barcodes.append(self.persistence(cloud, cloud,
                                          batch_size1 = batch_size_L,
                                          batch_size2 = 0,
                                          pdist_device = "cuda", is_plot = False))
    
            entropyes = self.entropy(barcodes, normalize=False)
            top_features[:, i:i+2] = entropyes.copy()
            
            mtd_features = [mtd.get_score(barc, 1, 'sum_length') for barc in barcodes]
            mtd_features = np.array(mtd_features).reshape(-1,1)
            top_features[:, ent_features+j] = mtd_features.copy().squeeze()
    
            mtd_features = [mtd.get_score(barc, 0, 'sum_length') for barc in barcodes]
            mtd_features = np.array(mtd_features).reshape(-1,1)
            top_features[:, ent_features+j+1] = mtd_features.copy().squeeze()
            pbar.update(1)
        
        # top_features[np.isnan(top_features)] = 0 ## I don't know where they appear from
        top_features = np.nan_to_num(top_features, posinf = 1000, neginf = -1000)
        top_features[top_features>3e30] = 3e30
        top_features[top_features<-3e30] = -3e30
        return top_features

    def inverse_transform(self):
        raise "Can't be implemented"
        
    def get_feature_names_out(self, input_features):
        names = []
        for index_repr in range(len(self.representatives)):
            for sufix in ["","rw_"]:
                names.append(f"entropyes_{sufix}repr_{index_repr}_dim_0")
                names.append(f"entropyes_{sufix}repr_{index_repr}_dim_1")
                
        for index_repr in range(len(self.representatives)):
            for sufix in ["","rw_"]:
                names.append(f"mtd_{sufix}repr_{index_repr}_dim_1")
                names.append(f"mtd_{sufix}repr_{index_repr}_dim_0")
        
        names = names + ["entropyes_solo_0", "entropyes_solo_1", "mtd_solo_1", "mtd_solo_0"]
        return names
        
    # def get_params(self, deep=True):
    #     return {"embedding_dimension": self.embedding_dimension,
    #             "embedding_time_delay": self.embedding_time_delay,
    #             "stride": self.stride,
    #             "n_components": self.n_components}

    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
    #     return self