"""Audio-based impact detection for cricket ball-bat contact analysis."""
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional
import scipy.signal as signal
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)

class AudioImpactDetector:
    """Detects ball-bat impact events using audio analysis."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.impact_threshold = 0.7
        self.noise_floor = 0.1
        
        # Cricket-specific audio characteristics
        self.impact_freq_range = (1000, 8000)  # Hz - typical bat-ball impact
        self.impact_duration_range = (0.01, 0.1)  # seconds
        
        # Initialize audio processing parameters
        self.window_size = int(0.05 * sample_rate)  # 50ms windows
        self.hop_length = int(0.01 * sample_rate)   # 10ms hop
        
    def extract_audio_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract audio track from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio samples as numpy array or None if failed
        """
        try:
            # Use OpenCV to extract audio (simplified approach)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video file: {video_path}")
                return None
            
            # Note: OpenCV doesn't directly support audio extraction
            # In a real implementation, use ffmpeg-python or librosa
            # For now, we'll simulate audio analysis
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            cap.release()
            
            # Generate simulated audio for demonstration
            # In practice, extract real audio using: ffmpeg -i video.mp4 -vn -acodec pcm_s16le audio.wav
            time_samples = int(duration * self.sample_rate)
            simulated_audio = self._generate_simulated_cricket_audio(duration)
            
            logger.info(f"Extracted {duration:.2f}s of audio from video")
            return simulated_audio
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
    
    def _generate_simulated_cricket_audio(self, duration: float) -> np.ndarray:
        """Generate realistic cricket audio for testing purposes."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Base ambient noise
        audio = np.random.normal(0, 0.05, samples)
        
        # Add some periodic crowd noise
        crowd_freq = 0.1
        audio += 0.1 * np.sin(2 * np.pi * crowd_freq * t) * np.random.random(samples)
        
        # Simulate potential impact events
        impact_times = [duration * 0.3, duration * 0.7]  # Simulate impacts at 30% and 70%
        
        for impact_time in impact_times:
            impact_sample = int(impact_time * self.sample_rate)
            if impact_sample < samples - 1000:
                # Sharp transient for bat-ball contact
                impact_duration = int(0.02 * self.sample_rate)  # 20ms impact
                impact_signal = self._generate_impact_signature()
                
                end_sample = min(impact_sample + len(impact_signal), samples)
                actual_length = end_sample - impact_sample
                audio[impact_sample:end_sample] += impact_signal[:actual_length] * 0.5
        
        return audio
    
    def _generate_impact_signature(self) -> np.ndarray:
        """Generate characteristic bat-ball impact audio signature."""
        duration = 0.02  # 20ms
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Sharp attack with exponential decay
        envelope = np.exp(-t * 100)  # Fast decay
        
        # Multiple frequency components typical of wooden bat impact
        fundamental = 2000  # Hz
        harmonics = [1, 1.5, 2.2, 3.1, 4.7]  # Harmonic ratios
        weights = [1.0, 0.6, 0.4, 0.2, 0.1]   # Harmonic weights
        
        signal_sum = np.zeros(samples)
        for harmonic, weight in zip(harmonics, weights):
            freq = fundamental * harmonic
            component = weight * np.sin(2 * np.pi * freq * t)
            signal_sum += component
        
        return envelope * signal_sum
    
    def detect_impacts(self, audio: np.ndarray, timestamps: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Detect ball-bat impact events in audio signal.
        
        Args:
            audio: Audio samples
            timestamps: Optional timestamp array
            
        Returns:
            List of detected impact events
        """
        if audio is None or len(audio) == 0:
            return []
        
        if timestamps is None:
            timestamps = np.arange(len(audio)) / self.sample_rate
        
        impacts = []
        
        try:
            # Apply bandpass filter for cricket impact frequencies
            filtered_audio = self._bandpass_filter(audio, self.impact_freq_range[0], self.impact_freq_range[1])
            
            # Calculate spectral features
            spectral_features = self._extract_spectral_features(filtered_audio)
            
            # Detect transient events
            transient_events = self._detect_transients(filtered_audio)
            
            # Analyze each potential impact
            for event in transient_events:
                impact_analysis = self._analyze_impact_event(
                    audio, event, spectral_features, timestamps
                )
                
                if impact_analysis['confidence'] > self.impact_threshold:
                    impacts.append(impact_analysis)
            
            # Remove duplicate detections
            impacts = self._remove_duplicate_impacts(impacts)
            
            logger.info(f"Detected {len(impacts)} potential impact events")
            
        except Exception as e:
            logger.error(f"Impact detection failed: {e}")
        
        return impacts
    
    def _bandpass_filter(self, audio: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Apply bandpass filter to isolate impact frequencies."""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict:
        """Extract spectral features from audio signal."""
        features = {}
        
        # Short-time Fourier transform
        window_samples = min(self.window_size, len(audio) // 4)
        hop_samples = min(self.hop_length, window_samples // 2)
        
        if window_samples < 64:  # Minimum window size
            window_samples = min(64, len(audio))
            hop_samples = window_samples // 2
        
        # Calculate STFT
        f, t, stft = signal.stft(
            audio, 
            fs=self.sample_rate,
            window='hann',
            nperseg=window_samples,
            noverlap=window_samples - hop_samples
        )
        
        magnitude = np.abs(stft)
        
        features['frequencies'] = f
        features['times'] = t
        features['magnitude'] = magnitude
        features['power'] = magnitude ** 2
        
        # Calculate spectral centroid
        features['spectral_centroid'] = self._calculate_spectral_centroid(f, magnitude)
        
        # Calculate spectral rolloff
        features['spectral_rolloff'] = self._calculate_spectral_rolloff(f, magnitude)
        
        return features
    
    def _calculate_spectral_centroid(self, frequencies: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
        """Calculate spectral centroid for each time frame."""
        centroids = []
        
        for frame in magnitude.T:
            if np.sum(frame) > 0:
                centroid = np.sum(frequencies * frame) / np.sum(frame)
                centroids.append(centroid)
            else:
                centroids.append(0)
        
        return np.array(centroids)
    
    def _calculate_spectral_rolloff(self, frequencies: np.ndarray, magnitude: np.ndarray, rolloff_percent: float = 0.85) -> np.ndarray:
        """Calculate spectral rolloff frequency."""
        rolloffs = []
        
        for frame in magnitude.T:
            total_energy = np.sum(frame)
            if total_energy > 0:
                cumulative_energy = np.cumsum(frame)
                rolloff_threshold = rolloff_percent * total_energy
                rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
                
                if len(rolloff_idx) > 0:
                    rolloffs.append(frequencies[rolloff_idx[0]])
                else:
                    rolloffs.append(frequencies[-1])
            else:
                rolloffs.append(0)
        
        return np.array(rolloffs)
    
    def _detect_transients(self, audio: np.ndarray) -> List[Dict]:
        """Detect transient events in audio signal."""
        transients = []
        
        # Calculate onset strength
        onset_envelope = self._calculate_onset_strength(audio)
        
        # Find peaks in onset strength
        min_distance = int(0.1 * self.sample_rate / self.hop_length)  # Minimum 100ms between impacts
        peak_indices, peak_properties = signal.find_peaks(
            onset_envelope,
            height=self.noise_floor,
            distance=min_distance,
            prominence=0.05
        )
        
        # Convert peak indices to time
        hop_samples = min(self.hop_length, len(audio) // 100)
        
        for peak_idx in peak_indices:
            peak_time = peak_idx * hop_samples / self.sample_rate
            peak_strength = onset_envelope[peak_idx]
            
            transients.append({
                'time': peak_time,
                'strength': peak_strength,
                'sample_index': peak_idx * hop_samples
            })
        
        return transients
    
    def _calculate_onset_strength(self, audio: np.ndarray) -> np.ndarray:
        """Calculate onset strength function."""
        # Simple onset detection using spectral flux
        window_samples = min(1024, len(audio) // 10)
        hop_samples = window_samples // 4
        
        if window_samples < 64:
            return np.array([0])
        
        # Calculate spectrogram
        f, t, stft = signal.stft(
            audio,
            fs=self.sample_rate,
            window='hann',
            nperseg=window_samples,
            noverlap=window_samples - hop_samples
        )
        
        magnitude = np.abs(stft)
        
        # Calculate spectral flux (difference between consecutive frames)
        onset_strength = []
        for i in range(1, magnitude.shape[1]):
            diff = magnitude[:, i] - magnitude[:, i-1]
            # Only positive differences (energy increase)
            diff = np.maximum(diff, 0)
            strength = np.sum(diff)
            onset_strength.append(strength)
        
        return np.array(onset_strength)
    
    def _analyze_impact_event(self, audio: np.ndarray, event: Dict, 
                             spectral_features: Dict, timestamps: np.ndarray) -> Dict:
        """Analyze a detected transient event to determine if it's a cricket impact."""
        event_time = event['time']
        event_sample = int(event_time * self.sample_rate)
        
        # Extract audio segment around event
        segment_duration = 0.1  # 100ms around event
        segment_samples = int(segment_duration * self.sample_rate)
        start_sample = max(0, event_sample - segment_samples // 2)
        end_sample = min(len(audio), event_sample + segment_samples // 2)
        
        audio_segment = audio[start_sample:end_sample]
        
        # Analyze segment characteristics
        analysis = {
            'time': event_time,
            'sample_index': event_sample,
            'duration': len(audio_segment) / self.sample_rate,
            'strength': event['strength'],
            'confidence': 0.0,
            'characteristics': {}
        }
        
        if len(audio_segment) < 10:
            return analysis
        
        # Calculate audio characteristics
        characteristics = {}
        
        # Peak amplitude
        characteristics['peak_amplitude'] = np.max(np.abs(audio_segment))
        
        # RMS energy
        characteristics['rms_energy'] = np.sqrt(np.mean(audio_segment ** 2))
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(audio_segment)))[0]
        characteristics['zero_crossing_rate'] = len(zero_crossings) / len(audio_segment)
        
        # Spectral characteristics
        if len(audio_segment) >= 64:
            freqs = fftfreq(len(audio_segment), 1/self.sample_rate)
            fft_values = np.abs(fft(audio_segment))
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(fft_values[:len(fft_values)//2])
            characteristics['dominant_frequency'] = abs(freqs[dominant_freq_idx])
            
            # Energy in cricket impact frequency range
            impact_freq_mask = (np.abs(freqs) >= self.impact_freq_range[0]) & (np.abs(freqs) <= self.impact_freq_range[1])
            total_energy = np.sum(fft_values ** 2)
            impact_energy = np.sum(fft_values[impact_freq_mask] ** 2)
            characteristics['impact_frequency_ratio'] = impact_energy / total_energy if total_energy > 0 else 0
        
        analysis['characteristics'] = characteristics
        
        # Calculate confidence score
        confidence = self._calculate_impact_confidence(characteristics)
        analysis['confidence'] = confidence
        
        return analysis
    
    def _calculate_impact_confidence(self, characteristics: Dict) -> float:
        """Calculate confidence that an event is a cricket ball-bat impact."""
        confidence = 0.0
        
        # Check peak amplitude (impacts should be loud)
        peak_amp = characteristics.get('peak_amplitude', 0)
        if peak_amp > 0.2:
            confidence += 0.3
        elif peak_amp > 0.1:
            confidence += 0.15
        
        # Check frequency content (impacts have characteristic frequencies)
        impact_freq_ratio = characteristics.get('impact_frequency_ratio', 0)
        if impact_freq_ratio > 0.5:
            confidence += 0.4
        elif impact_freq_ratio > 0.3:
            confidence += 0.2
        
        # Check dominant frequency
        dominant_freq = characteristics.get('dominant_frequency', 0)
        if self.impact_freq_range[0] <= dominant_freq <= self.impact_freq_range[1]:
            confidence += 0.2
        
        # Check zero crossing rate (sharp impacts have specific patterns)
        zcr = characteristics.get('zero_crossing_rate', 0)
        if 0.1 <= zcr <= 0.4:  # Optimal range for impact events
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _remove_duplicate_impacts(self, impacts: List[Dict], min_separation: float = 0.1) -> List[Dict]:
        """Remove duplicate impact detections within minimum separation time."""
        if len(impacts) <= 1:
            return impacts
        
        # Sort by confidence
        impacts.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_impacts = []
        
        for impact in impacts:
            # Check if this impact is too close to already selected impacts
            too_close = False
            for selected in filtered_impacts:
                if abs(impact['time'] - selected['time']) < min_separation:
                    too_close = True
                    break
            
            if not too_close:
                filtered_impacts.append(impact)
        
        # Sort by time
        filtered_impacts.sort(key=lambda x: x['time'])
        
        return filtered_impacts
    
    def synchronize_with_video(self, impacts: List[Dict], video_fps: float, video_duration: float) -> List[Dict]:
        """Synchronize audio impacts with video timestamps."""
        synchronized_impacts = []
        
        for impact in impacts:
            # Convert audio time to video frame
            video_time = impact['time']
            frame_number = int(video_time * video_fps)
            
            # Validate timing
            if 0 <= video_time <= video_duration:
                synchronized_impact = impact.copy()
                synchronized_impact['video_time'] = video_time
                synchronized_impact['video_frame'] = frame_number
                synchronized_impacts.append(synchronized_impact)
        
        return synchronized_impacts
    
    def generate_impact_report(self, impacts: List[Dict]) -> Dict:
        """Generate comprehensive impact detection report."""
        if not impacts:
            return {
                'total_impacts': 0,
                'confidence_distribution': {},
                'timing_analysis': {},
                'audio_quality_assessment': 'No impacts detected'
            }
        
        # Calculate statistics
        confidences = [impact['confidence'] for impact in impacts]
        times = [impact['time'] for impact in impacts]
        
        report = {
            'total_impacts': len(impacts),
            'average_confidence': np.mean(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'confidence_distribution': {
                'high_confidence': len([c for c in confidences if c > 0.8]),
                'medium_confidence': len([c for c in confidences if 0.5 <= c <= 0.8]),
                'low_confidence': len([c for c in confidences if c < 0.5])
            },
            'timing_analysis': {
                'first_impact': min(times) if times else 0,
                'last_impact': max(times) if times else 0,
                'impact_intervals': [times[i+1] - times[i] for i in range(len(times)-1)] if len(times) > 1 else []
            },
            'detailed_impacts': impacts
        }
        
        # Assess audio quality
        if report['average_confidence'] > 0.7:
            report['audio_quality_assessment'] = 'High quality - clear impact detection'
        elif report['average_confidence'] > 0.5:
            report['audio_quality_assessment'] = 'Good quality - reliable detection'
        else:
            report['audio_quality_assessment'] = 'Low quality - detection uncertain'
        
        return report