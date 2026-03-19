from vision_forge_api.config.loader import ConfigLoader
from vision_forge_api.catalog.service import TagCatalog
import json

loader = ConfigLoader('config')
catalog = TagCatalog(loader.load_tag_sets(), loader.load_profiles(), loader.load_prompts())
with open('data/embeddings/text_embeddings.json') as f:
    vectors = json.load(f)['vectors']
missing = set(catalog.canonical_tags()) - set(vectors.keys())
print('missing tags:', len(missing))