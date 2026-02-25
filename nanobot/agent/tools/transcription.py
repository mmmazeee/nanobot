"""Audio transcription tool using WhisperKit (macOS only)."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class TranscriptionTool(Tool):
    """Transcribe audio files to text using WhisperKit on macOS."""

    name = "transcribe"
    description = "Transcribe audio files to text using WhisperKit on macOS. Supports multiple formats (wav, mp3, m4a, flac, aac - native; ogg - via auto-conversion) and languages (auto, zh, en, ja, ko, es, fr, de, ru, etc.). OGG format is auto-converted to WAV using ffmpeg for Discord audio files."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the audio file to transcribe"
            },
            "language": {
                "type": "string",
                "description": "Language code (auto for automatic detection, or specific codes like zh, en, ja, ko, es, fr, de, ru)",
                "default": "auto"
            },
            "model": {
                "type": "string",
                "description": "Model size (large-v3 for best accuracy, medium, small, tiny, base for faster speed)",
                "default": "large-v3"
            }
        },
        "required": ["file_path"]
    }

    # Supported audio formats (WhisperKit native support)
    ALLOWED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".aac"}
    
    # Formats requiring conversion (requires ffmpeg)
    CONVERTIBLE_FORMATS = {".ogg"}
    
    # Combined: all formats we can handle
    SUPPORTED_FORMATS = ALLOWED_FORMATS | CONVERTIBLE_FORMATS
    
    # Available WhisperKit models
    AVAILABLE_MODELS = {"large-v3", "medium", "small", "tiny", "base"}
    
    # Common language codes
    AVAILABLE_LANGUAGES = {"auto", "zh", "en", "ja", "ko", "es", "fr", "de", "ru", "it", "pt", "nl", "tr", "pl", "ar", "vi", "th", "sv", "no", "da", "fi"}

    def __init__(self, config: Any = None):
        """
        Initialize TranscriptionTool.
        
        Args:
            config: TranscriptionConfig object with enabled, model, language, allowed_formats
        """
        self.config = config
        if config:
            self.default_model = config.model or "large-v3"
            self.default_language = config.language or "auto"
            self.allowed_formats = set(config.allowed_formats) if config.allowed_formats else self.ALLOWED_FORMATS
            self.subagent_threshold = getattr(config, 'subagent_threshold', 60)
            self.enable_auto_delegate = getattr(config, 'enable_auto_delegate', True)
        else:
            self.default_model = "large-v3"
            self.default_language = "auto"
            self.allowed_formats = self.ALLOWED_FORMATS
            self.subagent_threshold = 60
            self.enable_auto_delegate = True
        
        self._spawn_tool = None
        self._channel = None
        self._chat_id = None

    def set_spawn_tool(self, spawn_tool: Any) -> None:
        """Inject SpawnTool for delegation support."""
        self._spawn_tool = spawn_tool
    
    def set_context(self, channel: str, chat_id: str) -> None:
        """Set channel context for notifications."""
        self._channel = channel
        self._chat_id = chat_id
        # Sync spawn_tool context to ensure notifications are sent to the right place
        if self._spawn_tool:
            self._spawn_tool.set_context(channel, chat_id)

    async def execute(self, file_path: str, language: str | None = None, model: str | None = None, **kwargs: Any) -> str:
        """
        Execute transcription of an audio file.
        
        Args:
            file_path: Path to the audio file
            language: Language code (optional, uses config default if not specified)
            model: Model size (optional, uses config default if not specified)
        
        Returns:
            Transcribed text or error message
        """
        # Check platform
        if sys.platform != "darwin":
            return "Error: WhisperKit transcription is only available on macOS. Please ensure you are running on a Mac with Apple Silicon."
        
        # Normalize and validate file path
        try:
            audio_path = Path(file_path).expanduser().resolve()
        except Exception as e:
            return f"Error: Invalid file path '{file_path}': {e}"
        
        # Check if file exists
        if not audio_path.exists():
            return f"Error: File not found: {audio_path}"
        
        # Check if it's a file (not a directory)
        if not audio_path.is_file():
            return f"Error: Path is not a file: {audio_path}"
        
        # Validate file format (check against all supported formats, including convertible ones)
        file_ext = audio_path.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            return f"Error: Unsupported file format '{file_ext}'. Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
        
        # Check audio duration for delegation
        duration = await self._get_audio_duration(audio_path)
        if duration is None:
            logger.warning("Could not detect audio duration, proceeding with direct transcription")
        elif self._should_use_subagent(duration):
            logger.info("Audio duration {}s exceeds threshold {}s, delegating to subagent", duration, self.subagent_threshold)
            return await self._delegate_to_subagent(file_path, duration, language, model)
        
        # Direct transcription for short audio
        temp_file = None
        try:
            if file_ext in self.CONVERTIBLE_FORMATS:
                # Convert to temporary WAV file
                converted_path = await self._convert_audio(audio_path)
                if converted_path is None:
                    return f"Error: Failed to convert '{file_ext}' format. Please install ffmpeg: brew install ffmpeg"
                temp_file = converted_path
                audio_path = converted_path
                logger.info("Converted {} to temporary WAV: {}", file_path, converted_path)
            
            # Validate language
            lang = language or self.default_language
            if lang not in self.AVAILABLE_LANGUAGES:
                return f"Error: Unsupported language '{lang}'. Supported: {', '.join(sorted(self.AVAILABLE_LANGUAGES))}"
            
            # Validate model
            model_size = model or self.default_model
            if model_size not in self.AVAILABLE_MODELS:
                return f"Error: Unsupported model '{model_size}'. Supported: {', '.join(sorted(self.AVAILABLE_MODELS))}"
            
            # Check if whisperkit-cli is available
            if not self._check_whisperkit_available():
                return "Error: whisperkit-cli not found. Please install it with: brew install whisperkit"
            
            # Execute transcription
            result_text = await self._transcribe(audio_path, lang, model_size)
            return result_text
        except asyncio.TimeoutError:
            return f"Error: Transcription timed out after 5 minutes. The file may be too long or there may be an issue with WhisperKit."
        except Exception as e:
            logger.exception("Transcription failed")
            return f"Error: Transcription failed: {e}"
        finally:
            # Cleanup temporary file
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.debug("Cleaned up temporary file: {}", temp_file)
                except Exception as e:
                    logger.warning("Failed to cleanup temporary file {}: {}", temp_file, e)

    def _check_whisperkit_available(self) -> bool:
        """Check if whisperkit-cli command is available."""
        try:
            import shutil
            return shutil.which("whisperkit-cli") is not None
        except Exception:
            return False

    async def _transcribe(self, audio_path: Path, language: str, model: str) -> str:
        """
        Execute WhisperKit transcription asynchronously.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            model: Model size
        
        Returns:
            Transcribed text
        
        Raises:
            asyncio.TimeoutError: If transcription takes too long
            Exception: If transcription fails
        """
        # Build command
        cmd = [
            "whisperkit-cli",
            "transcribe",
            "--audio-path", str(audio_path),
            "--model", model,
            "--language", language
        ]
        
        logger.info("Running WhisperKit transcription: {}", " ".join(cmd))
        
        # Run subprocess with timeout
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for completion with 5-minute timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300.0  # 5 minutes
            )
            
            # Check return code
            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                return f"Error: WhisperKit failed with code {process.returncode}. {error_msg}"
            
            # Parse output
            output = stdout.decode("utf-8", errors="replace").strip()
            
            if not output:
                return "Warning: No transcription output generated. The audio may be silent or too short."
            
            # Format result
            return f"Transcription completed:\n\n{output}"
            
        except asyncio.TimeoutError:
            # Kill process if timeout
            try:
                if process:
                    process.kill()
                    await process.wait()
            except Exception:
                pass
            raise

    def _check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg command is available."""
        try:
            import shutil
            return shutil.which("ffmpeg") is not None
        except Exception:
            return False
    
    async def _get_audio_duration(self, audio_path: Path) -> float | None:
        """
        Get audio duration in seconds using ffprobe.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Duration in seconds, or None if detection failed
        """
        try:
            import shutil
            ffprobe_path = shutil.which("ffprobe")
            if not ffprobe_path:
                logger.warning("ffprobe not found, cannot detect audio duration")
                return None
        except Exception:
            logger.warning("Failed to check for ffprobe")
            return None
        
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=10.0
            )
            
            if process.returncode != 0:
                logger.warning("ffprobe failed: {}", stderr.decode("utf-8", errors="replace"))
                return None
            
            duration_str = stdout.decode("utf-8", errors="replace").strip()
            if not duration_str:
                return None
            
            return float(duration_str)
            
        except asyncio.TimeoutError:
            logger.warning("ffprobe timed out")
            return None
        except Exception as e:
            logger.warning("ffprobe failed to get duration: {}", e)
            return None
    
    def _should_use_subagent(self, duration: float) -> bool:
        """
        Check if audio should be delegated to subagent based on duration.
        
        Args:
            duration: Audio duration in seconds
        
        Returns:
            True if should delegate, False otherwise
        """
        if not self.enable_auto_delegate:
            return False
        if not self._spawn_tool:
            logger.warning("SpawnTool not available, cannot delegate to subagent")
            return False
        return duration > self.subagent_threshold
    
    async def _delegate_to_subagent(
        self,
        file_path: str,
        duration: float,
        language: str | None = None,
        model: str | None = None
    ) -> str:
        """
        Delegate transcription to subagent for long audio.

        Args:
            file_path: Path to audio file
            duration: Audio duration in seconds
            language: Language code
            model: Model size

        Returns:
            Message indicating delegation
        """
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_str = f"{minutes}分{seconds}秒" if minutes > 0 else f"{seconds}秒"

        lang = language or self.default_language
        model_size = model or self.default_model

        task_desc = f"请使用 transcribe 工具转录音频文件：{file_path}\n"
        task_desc += f"- 语言：{lang}\n"
        task_desc += f"- 模型：{model_size}\n"
        task_desc += f"- 时长：{duration_str}\n\n"
        task_desc += "请完成转录后，直接返回完整的转录结果。"

        label = f"转录音频 ({duration_str})"

        # Log context for debugging
        logger.info("Delegating transcription: file={}, duration={}, target={}:{}, spawn_context={}:{}",
                   file_path, duration_str,
                   self._channel, self._chat_id,
                   self._spawn_tool._origin_channel if self._spawn_tool else "None",
                   self._spawn_tool._origin_chat_id if self._spawn_tool else "None")

        try:
            result = await self._spawn_tool.execute(task=task_desc, label=label)
            logger.info("Subagent delegation successful: {}", result)
            return f"✅ 音频文件已提交给后台助手进行转录（时长：{duration_str}）。\n\n转录完成后，我会立即通知您并提供完整结果。"
        except Exception as e:
            logger.exception("Failed to delegate to subagent")
            return f"❌ 转录委派失败: {e}"

    async def _convert_audio(self, input_path: Path) -> Path | None:
        """
        Convert audio file to WAV format using ffmpeg.
        
        Args:
            input_path: Path to input audio file
        
        Returns:
            Path to temporary WAV file, or None if conversion failed
        """
        if not self._check_ffmpeg_available():
            logger.error("ffmpeg not found for audio conversion")
            return None
        
        # Create temporary file
        import tempfile
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="whisperkit_")
            temp_file = Path(temp_path)
            # Close the file descriptor since we'll use subprocess
            import os
            os.close(temp_fd)
        except Exception as e:
            logger.error("Failed to create temporary file: {}", e)
            return None
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(input_path),  # Input file
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",  # 16kHz sample rate (Whisper standard)
            "-ac", "1",  # Mono
            "-y",  # Overwrite output
            str(temp_file)  # Output file
        ]
        
        logger.info("Converting {} to WAV: {}", input_path, " ".join(cmd))
        
        # Run conversion
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for completion (30 second timeout)
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60.0  # 60 seconds for conversion
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                logger.error("ffmpeg conversion failed: {}", error_msg)
                # Cleanup temp file
                if temp_file.exists():
                    temp_file.unlink()
                return None
            
            if not temp_file.exists():
                logger.error("ffmpeg completed but output file not created")
                return None
            
            logger.info("Conversion successful: {} -> {}", input_path, temp_file)
            return temp_file
            
        except asyncio.TimeoutError:
            logger.error("ffmpeg conversion timed out")
            if temp_file.exists():
                temp_file.unlink()
            return None
        except Exception as e:
            logger.exception("ffmpeg conversion failed")
            if temp_file.exists():
                temp_file.unlink()
            return None
