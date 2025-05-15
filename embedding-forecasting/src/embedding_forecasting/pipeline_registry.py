from kedro.framework.session import KedroSession
from .pipelines.data_processing.pipeline import create_pipeline as dp
from .pipelines.cnn_autoencoder.pipeline import create_pipeline as cnn
from .pipelines.pca_embedding.pipeline import create_pipeline as embedding
from .pipelines.lle_embedding.pipeline import create_pipeline as lle_embedding
from .pipelines.umap_embedding.pipeline import create_pipeline as umap_embedding
from .pipelines.wavelet_embedding.pipeline import create_pipeline as wavelet_embedding
from .pipelines.fft_embedding.pipeline import create_pipeline as fft_embedding

def register_pipelines():
   session = KedroSession.create()
   context = session.load_context()
   params = context.params

   dp_pipe  = dp(**params)
   cnn_pipe = cnn(**params)
   emb_pipe = embedding(**params)
   lle_emb_pipe = lle_embedding(**params)
   umap_emb_pipe = umap_embedding(**params)
   wavelet_emb_pipe = wavelet_embedding(**params)
   fft_emb_pipe = fft_embedding(**params)

   return {
     "data_processing": dp_pipe,
     "cnn_autoencoder": cnn_pipe,
     "pca_embedding": emb_pipe,
     "lle_embedding": lle_emb_pipe,
     "umap_embedding": umap_emb_pipe,
     "wavelet_embedding": wavelet_emb_pipe,
     "fft_embedding": fft_emb_pipe,
     "__default__": dp_pipe + cnn_pipe + emb_pipe + lle_emb_pipe + umap_emb_pipe + wavelet_emb_pipe + fft_emb_pipe,
   }