from vision_forge_api.config.loader import ConfigLoader
from vision_forge_api.catalog.service import TagCatalog
from vision_forge_api.siglip.service import SiglipService
from vision_forge_api.predict.service import PredictionService
from PIL import Image
from pathlib import Path

loader = ConfigLoader("/config")
settings = loader.load_settings()
catalog = TagCatalog(
    loader.load_tag_sets(), loader.load_profiles(), loader.load_prompts()
)
siglip = SiglipService(settings.siglip_model_id, settings.model_cache_dir)
pred = PredictionService(catalog, siglip, settings.embeddings_dir)
image = Image.open("/samples/animal_dog.jpg").convert("RGB")
profile = catalog.profile_detail("default")
preds = pred.score_image(image, profile.canonical_tags, [], 0.0, 10)
print("canonical tags", len(profile.canonical_tags))
print(
    "vectors available",
    sum(1 for tag in profile.canonical_tags if pred.embedding_for_tag(tag) is not None),
)
print("pred count", len(preds))
for result in preds[:5]:
    print(result.canonical_tag, result.score)
