from kedro.framework.session import KedroSession
from .pipelines.data_processing.pipeline import create_pipeline as dp
from .pipelines.cnn_autoencoder.pipeline import create_pipeline as cnn
from .pipelines.pca_embedding.pipeline import create_pipeline as embedding
from .pipelines.lle_embedding.pipeline import create_pipeline as lle_embedding
from .pipelines.umap_embedding.pipeline import create_pipeline as umap_embedding
from .pipelines.wavelet_embedding.pipeline import create_pipeline as wavelet_embedding
from .pipelines.fft_embedding.pipeline import create_pipeline as fft_embedding
from .pipelines.graph_embedding.pipeline import create_pipeline as graph_embedding
from .pipelines.tda_embedding.pipeline import create_pipeline as tda_embedding
from .pipelines.gaf_imaging.pipeline import create_pipeline as gaf_imaging
from .pipelines.mtf_imaging.pipeline import create_pipeline as mtf_imaging
from .pipelines.rp_imaging.pipeline import create_pipeline as rp_imaging
from .pipelines.clip_embedding.pipeline import create_pipeline as clip_embedding
from .pipelines.dino_embedding.pipeline import create_pipeline as dino_embedding
from .pipelines.resnet_embedding.pipeline import create_pipeline as resnet_embedding
from .pipelines.volatility_classification.pipeline import create_pipeline as volatility_classification
from .pipelines.fusion_volatility_classification.pipeline import create_pipeline as fusion_volatility_classification



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
   graph_emb_pipe = graph_embedding(**params)
   tda_emb_pipe = tda_embedding(**params)
   gaf_img_pipe = gaf_imaging(**params)
   mtf_img_pipe = mtf_imaging(**params)
   rp_img_pipe = rp_imaging(**params)
   clip_emb_pipe = clip_embedding(**params)
   dino_emb_pipe = dino_embedding(**params)
   resnet_emb_pipe = resnet_embedding(**params)
   vol_class_pipe = volatility_classification(**params)
   fusion_vol_class_pipe = fusion_volatility_classification(**params)

   return {
     "data_processing": dp_pipe,
     "cnn_autoencoder": cnn_pipe,
     "pca_embedding": emb_pipe,
     "lle_embedding": lle_emb_pipe,
     "umap_embedding": umap_emb_pipe,
     "wavelet_embedding": wavelet_emb_pipe,
     "fft_embedding": fft_emb_pipe,
     "graph_embedding": graph_emb_pipe,
     "tda_embedding": tda_emb_pipe,
     "gaf_imaging": gaf_img_pipe,
     "mtf_imaging": mtf_img_pipe,
     "rp_imaging": rp_img_pipe,
     "clip_embedding": clip_emb_pipe,
     "dino_embedding": dino_emb_pipe,
     "resnet_embedding": resnet_emb_pipe,
     "volatility_classification": vol_class_pipe,
     "fusion_volatility_classification": fusion_vol_class_pipe,
     "__default__": dp_pipe + cnn_pipe + emb_pipe + lle_emb_pipe + umap_emb_pipe + wavelet_emb_pipe + fft_emb_pipe + graph_emb_pipe + tda_emb_pipe + gaf_img_pipe + mtf_img_pipe + rp_img_pipe,
   }