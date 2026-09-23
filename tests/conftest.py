import os
import tempfile

# isolate experiment memory before epoch.config.settings() is first cached
os.environ.setdefault("EPOCH_HOME", tempfile.mkdtemp(prefix="epoch-test-"))
os.environ["EPOCH_OFFLINE"] = "1"
