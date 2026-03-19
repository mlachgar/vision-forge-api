import torch
from collections import OrderedDict
from vision_forge_api.config.loader import ConfigLoader
from vision_forge_api.catalog.service import TagCatalog
from vision_forge_api.siglip.service import SiglipService
from vision_forge_api.predict.service import PredictionService
from PIL import Image
loader = ConfigLoader('/config')
settings = loader.load_settings()
catalog = TagCatalog(loader.load_tag_sets(), loader.load_profiles(), loader.load_prompts())
siglip = SiglipService(settings.siglip_model_id, settings.model_cache_dir)
pred = PredictionService(catalog, siglip, settings.embeddings_dir)
image = Image.open('/samples/animal_dog.jpg').convert('RGB')
profile = catalog.profile_detail('default')
image_vector = siglip.encode_image(image)
can_keys = tuple(OrderedDict.fromkeys(profile.canonical_tags))
vectors = []
labels = []
for key in can_keys:
    vector = pred.embedding_for_tag(key)
    if vector is not None:
        vectors.append(vector)
        labels.append(key)
print('candidate count', len(vectors))
if not vectors:
    raise SystemExit('no vectors')
candidate_tensor = torch.stack(vectors, dim=0)
scores = torch.matmul(candidate_tensor, image_vector.T).squeeze(-1)
print('scores min', float(scores.min()), 'max', float(scores.max()))
print('scores[:10]', [float(s) for s in scores.tolist()[:10]])
cat_vec = pred.embedding_for_tag('cat')
print('cat score', float(torch.matmul(cat_vec, image_vector.T)))
