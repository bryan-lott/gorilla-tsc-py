import os

from hypothesis import settings

settings.register_profile("mutants", max_examples=25, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
