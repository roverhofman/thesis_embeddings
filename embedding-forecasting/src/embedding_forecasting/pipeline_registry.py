from kedro.framework.session import KedroSession
from .pipelines.data_processing.pipeline import create_pipeline as dp
from .pipelines.cnn_autoencoder.pipeline import create_pipeline as cnn
from .pipelines.pca_embedding.pipeline import create_pipeline as embedding

def register_pipelines():
   session = KedroSession.create()
   context = session.load_context()
   params = context.params

   dp_pipe  = dp(**params)
   cnn_pipe = cnn(**params)
   emb_pipe = embedding(**params)

   return {
     "data_processing": dp_pipe,
     "cnn_autoencoder": cnn_pipe,
     "pca_embedding": emb_pipe,
     "__default__": dp_pipe + cnn_pipe + emb_pipe,
   }