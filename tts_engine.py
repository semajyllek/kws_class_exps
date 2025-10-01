"""
TTS engine management for synthetic data generation.
"""
import logging

logger = logging.getLogger(__name__)

# Try to import Bark
try:
    from bark import SAMPLE_RATE as BARK_SAMPLE_RATE, generate_audio, preload_models
    import scipy.io.wavfile
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    logger.warning("Bark not available. Install with: pip install git+https://github.com/suno-ai/bark.git")


class TTSEngine:
    """Manages different TTS engines (gTTS and Bark)"""
    
    def __init__(self, engine_type: str = 'gtts'):
        """
        Initialize TTS engine.
        
        Args:
            engine_type: 'gtts' (fast, robotic) or 'bark' (slow, natural)
        """
        self.engine_type = engine_type
        self.bark_initialized = False
        
        if engine_type == 'bark':
            if not BARK_AVAILABLE:
                raise RuntimeError(
                    "Bark not installed. Install with:\n"
                    "pip install git+https://github.com/suno-ai/bark.git"
                )
            logger.info("Initializing Bark TTS (first run will download models ~2GB)")
            logger.info("This may take 2-5 minutes on first use...")
            
            # Preload models (downloads on first run)
            preload_models()
            self.bark_initialized = True
            
            logger.info("Bark TTS initialized successfully!")
        else:
            logger.info("Using gTTS engine")
    
    def synthesize(self, text: str, output_path: str):
        """
        Synthesize text to audio file.
        
        Args:
            text: Text to synthesize
            output_path: Path to save WAV file
        """
        if self.engine_type == 'gtts':
            self._synthesize_gtts(text, output_path)
        elif self.engine_type == 'bark':
            self._synthesize_bark(text, output_path)
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine_type}")
    
    def _synthesize_gtts(self, text: str, output_path: str):
        """Synthesize using gTTS"""
        from gtts import gTTS
        
        slow = '...' in text or len(text) > 8
        tts = gTTS(text=text, lang='en', slow=slow)
        tts.save(output_path)
    
    def _synthesize_bark(self, text: str, output_path: str):
        """Synthesize using Bark"""
        # Clean text - Bark handles punctuation naturally
        clean_text = text.strip()
        
        # Bark uses special markers for prosody
        if text.isupper():
            clean_text = f"[LOUD] {clean_text.lower()}"
        elif '...' in text:
            clean_text = clean_text.replace('...', '...')  # Bark handles pauses
        
        try:
            # Generate audio with Bark
            # Use voice preset for consistency (v2/en_speaker_6 is clear and neutral)
            audio_array = generate_audio(
                clean_text,
                history_prompt="v2/en_speaker_6"  # Consistent neutral English voice
            )
            
            # Bark outputs at 24kHz, save as wav
            scipy.io.wavfile.write(
                output_path,
                rate=BARK_SAMPLE_RATE,
                data=audio_array
            )
            
        except Exception as e:
            # Fallback: try without voice preset
            logger.warning(f"Bark synthesis failed with preset, trying default: {e}")
            audio_array = generate_audio(clean_text)
            scipy.io.wavfile.write(
                output_path,
                rate=BARK_SAMPLE_RATE,
                data=audio_array
            )
