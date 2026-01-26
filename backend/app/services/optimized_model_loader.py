import logging
import asyncio
from typing import Optional, Dict, Any
from functools import lru_cache
import time
from app.core.config import settings

logger = logging.getLogger(__name__)

# Global flag for AI availability
AI_AVAILABLE = False
LIGHTWEIGHT_AI_AVAILABLE = False

try:
    import torch
    from transformers import pipeline
    AI_AVAILABLE = True
    logger.info("[SUCCESS] Full AI capabilities available (PyTorch + Transformers)")
except ImportError:
    logger.warning("[ERROR] PyTorch or Transformers not available. Checking for lightweight alternatives...")

# Check for lightweight alternatives
try:
    from sentence_transformers import SentenceTransformer
    LIGHTWEIGHT_AI_AVAILABLE = True
    logger.info("[SUCCESS] Lightweight AI capabilities available (sentence-transformers)")
except ImportError:
    logger.warning("[ERROR] No AI libraries available. Using rule-based processing only.")

class OptimizedModelEngine:
    """
    Optimized model engine with multiple loading strategies and caching
    """
    
    def __init__(self):
        self._models = {}  # Model cache
        self._loading_locks = {}  # Prevent concurrent loading of same model
        self._model_stats = {}  # Track usage and performance
        
    @lru_cache(maxsize=1)
    def _get_device_info(self):
        """Get optimal device configuration with caching"""
        if not AI_AVAILABLE:
            return -1, "cpu"
        
        try:
            if torch.cuda.is_available() and settings.MODEL_DEVICE != -1:
                device = 0
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"[INIT] Using GPU: {device_name}")
                return device, "cuda"
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
        
        logger.info("[CONFIG] Using CPU for model processing")
        return -1, "cpu"
    
    async def get_lightweight_summarizer(self):
        """
        Get lightweight summarizer optimized for speed
        Priority: DistilBART -> T5-small -> Extractive methods
        """
        model_key = "lightweight_summarizer"
        
        if model_key in self._models:
            return self._models[model_key]
        
        # Prevent concurrent loading
        if model_key in self._loading_locks:
            while model_key in self._loading_locks:
                await asyncio.sleep(0.1)
            return self._models.get(model_key)
        
        self._loading_locks[model_key] = True
        
        try:
            logger.info("[LOADING] Loading lightweight summarizer...")
            start_time = time.time()
            
            device, device_type = self._get_device_info()
            
            if AI_AVAILABLE:
                # Try models in order of speed vs quality trade-off
                model_options = [
                    ("sshleifer/distilbart-cnn-12-6", "DistilBART (fast)"),
                    ("t5-small", "T5-Small (very fast)"),
                    ("facebook/bart-large-cnn", "BART-Large (fallback)")
                ]
                
                for model_name, description in model_options:
                    try:
                        logger.info(f"[ATTEMPTING] Attempting to load {description}...")
                        
                        summarizer = pipeline(
                            "summarization",
                            model=model_name,
                            device=device,
                            model_kwargs={"torch_dtype": torch.float16} if device_type == "cuda" else {}
                        )
                        
                        load_time = time.time() - start_time
                        logger.info(f"[SUCCESS] Loaded {description} in {load_time:.1f}s")
                        
                        self._models[model_key] = summarizer
                        self._model_stats[model_key] = {
                            "model": model_name,
                            "load_time": load_time,
                            "device": device_type,
                            "usage_count": 0
                        }
                        
                        return summarizer
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {description}: {e}")
                        continue
                
                logger.error("Failed to load any lightweight summarization model")
                return None
            else:
                logger.info("[CONFIG] AI not available - using rule-based processing")
                return None
                
        finally:
            del self._loading_locks[model_key]
    
    async def get_full_summarizer(self):
        """
        Get full-quality summarizer (original BART model)
        """
        model_key = "full_summarizer"
        
        if model_key in self._models:
            return self._models[model_key]
        
        # Prevent concurrent loading
        if model_key in self._loading_locks:
            while model_key in self._loading_locks:
                await asyncio.sleep(0.1)
            return self._models.get(model_key)
        
        self._loading_locks[model_key] = True
        
        try:
            logger.info("[LOADING] Loading full-quality summarizer...")
            start_time = time.time()
            
            if not AI_AVAILABLE:
                logger.warning("AI models not available")
                return None
            
            device, device_type = self._get_device_info()
            
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device,
                model_kwargs={"torch_dtype": torch.float16} if device_type == "cuda" else {}
            )
            
            load_time = time.time() - start_time
            logger.info(f"[SUCCESS] Loaded BART-Large-CNN in {load_time:.1f}s")
            
            self._models[model_key] = summarizer
            self._model_stats[model_key] = {
                "model": "facebook/bart-large-cnn",
                "load_time": load_time,
                "device": device_type,
                "usage_count": 0
            }
            
            return summarizer
            
        finally:
            del self._loading_locks[model_key]
    
    async def get_longformer_summarizer(self):
        """
        Get Longformer model for long-document understanding (4096 tokens)
        Ideal for research papers with long methodology, results sections
        """
        model_key = "longformer_summarizer"
        
        if model_key in self._models:
            return self._models[model_key]
        
        # Prevent concurrent loading
        if model_key in self._loading_locks:
            while model_key in self._loading_locks:
                await asyncio.sleep(0.1)
            return self._models.get(model_key)
        
        self._loading_locks[model_key] = True
        
        try:
            logger.info("[LOADING] Loading Longformer for long-document analysis...")
            start_time = time.time()
            
            if not AI_AVAILABLE:
                logger.warning("AI models not available")
                return None
            
            device, device_type = self._get_device_info()
            
            # Try LED (Longformer-Encoder-Decoder) which is better for summarization
            model_options = [
                ("allenai/led-base-16384", "LED-Base (16384 tokens)"),
                ("pszemraj/led-base-book-summary", "LED-Base fine-tuned")
            ]
            
            for model_name, description in model_options:
                try:
                    logger.info(f"[ATTEMPTING] Attempting to load {description}...")
                    
                    summarizer = pipeline(
                        "summarization",
                        model=model_name,
                        device=device,
                        model_kwargs={"torch_dtype": torch.float16} if device_type == "cuda" else {}
                    )
                    
                    load_time = time.time() - start_time
                    logger.info(f"[SUCCESS] Loaded {description} in {load_time:.1f}s")
                    
                    self._models[model_key] = summarizer
                    self._model_stats[model_key] = {
                        "model": model_name,
                        "load_time": load_time,
                        "device": device_type,
                        "usage_count": 0,
                        "max_tokens": 16384
                    }
                    
                    return summarizer
                    
                except Exception as e:
                    logger.warning(f"Failed to load {description}: {e}")
                    continue
            
            logger.warning("Failed to load Longformer, will use BART for long documents")
            return None
            
        finally:
            del self._loading_locks[model_key]
    
    @lru_cache(maxsize=1)
    async def get_sentence_transformer(self):
        """
        Get sentence transformer for embeddings (cached)
        """
        if not LIGHTWEIGHT_AI_AVAILABLE:
            return None
        
        model_key = "sentence_transformer"
        
        if model_key in self._models:
            return self._models[model_key]
        
        try:
            logger.info("[LOADING] Loading sentence transformer...")
            start_time = time.time()
            
            # Use lightweight but effective model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            load_time = time.time() - start_time
            logger.info(f"[SUCCESS] Loaded SentenceTransformer in {load_time:.1f}s")
            
            self._models[model_key] = model
            self._model_stats[model_key] = {
                "model": "all-MiniLM-L6-v2",
                "load_time": load_time,
                "usage_count": 0
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            return None
    
    def preload_models_async(self):
        """
        Preload models in background to reduce first-request latency
        """
        async def _preload():
            logger.info("[STARTING] Starting background model preloading...")
            try:
                # Load lightweight model only (DistilBART)
                # BART-Large-CNN and Longformer skipped due to high memory requirements
                await self.get_lightweight_summarizer()
                
                logger.info("[INFO] BART-Large-CNN and Longformer available on-demand (not preloaded to save memory)")
                logger.info("[SUCCESS] Background model preloading completed")
            except Exception as e:
                logger.error(f"[ERROR] Background model preloading failed: {e}")
                
                # If resources allow, preload full model too
                if settings.ENABLE_AI_MODELS:
                    await self.get_full_summarizer()
                    
                logger.info("[SUCCESS] Background model preloading completed")
            except Exception as e:
                logger.error(f"Background model preloading failed: {e}")
        
        # Run in background
        asyncio.create_task(_preload())
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all loaded models"""
        stats = {
            "loaded_models": len(self._models),
            "total_memory_usage": self._estimate_memory_usage(),
            "ai_available": AI_AVAILABLE,
            "lightweight_ai_available": LIGHTWEIGHT_AI_AVAILABLE,
            "models": self._model_stats.copy()
        }
        return stats
    
    def _estimate_memory_usage(self) -> str:
        """Estimate total memory usage of loaded models"""
        if not AI_AVAILABLE:
            return "0 MB"
        
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                return f"{gpu_memory:.1f} MB (GPU)"
            else:
                # Rough estimation for CPU
                model_count = len(self._models)
                estimated_mb = model_count * 500  # ~500MB per BART model
                return f"~{estimated_mb} MB (CPU)"
        except:
            return "Unknown"
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        logger.info("[CLEANUP] Clearing model cache...")
        
        for model_key in list(self._models.keys()):
            try:
                model = self._models[model_key]
                if hasattr(model, 'model') and hasattr(model.model, 'cpu'):
                    model.model.cpu()  # Move to CPU before deletion
                del self._models[model_key]
            except Exception as e:
                logger.warning(f"Error clearing model {model_key}: {e}")
        
        self._models.clear()
        self._model_stats.clear()
        
        # Clear PyTorch cache if available
        if AI_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("[SUCCESS] Model cache cleared")

# Global instance
optimized_model_engine = OptimizedModelEngine()

# Preload models on startup
def init_models():
    """Initialize models on application startup"""
    if settings.ENABLE_AI_MODELS:
        optimized_model_engine.preload_models_async()