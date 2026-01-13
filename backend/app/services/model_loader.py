import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Global flag for AI availability
AI_AVAILABLE = False
try:
    import torch
    from transformers import pipeline
    AI_AVAILABLE = True
except ImportError:
    logger.warning("Torch or Transformers not available. AI features will be disabled.")

class ModelEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelEngine, cls).__new__(cls)
            cls._instance.summarizer = None
            cls._instance.loaded = False
        return cls._instance

    def load_models(self):
        if self.loaded:
            return

        if not settings.ENABLE_AI_MODELS:
            logger.info("AI Models disabled by settings")
            return

        if not AI_AVAILABLE:
            logger.warning("AI dependencies missing. Skipping model load.")
            return

        try:
            logger.info("Loading BART summarization model...")
            device = 0 if torch.cuda.is_available() and settings.MODEL_DEVICE != -1 else -1
            
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
            self.loaded = True
            logger.info(f"BART model loaded successfully on device {device}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.loaded = False
            # robustly continue without crashing


    def get_summarizer(self):
        if not self.loaded:
            self.load_models()
        return self.summarizer

model_engine = ModelEngine()
