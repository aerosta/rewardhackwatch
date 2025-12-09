"""Tests for ML detector module."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from rewardhackwatch.core.detectors.base import RiskLevel
from rewardhackwatch.core.detectors.ml_detector import (
    CoTClassifierModel,
    MLDetector,
)


class TestCoTClassifierModel:
    """Tests for CoTClassifierModel."""

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_initialization(self, mock_auto_model):
        """Test model initialization."""
        # Mock encoder config
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        model = CoTClassifierModel()
        assert model.dropout is not None
        assert model.hack_classifier is not None
        assert model.misalign_classifier is not None

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_forward_pass(self, mock_auto_model):
        """Test forward pass produces correct output shape."""
        # Mock encoder
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768

        # Mock output
        batch_size = 2
        seq_len = 10
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, 768)
        mock_encoder.return_value = mock_output

        mock_auto_model.from_pretrained.return_value = mock_encoder

        model = CoTClassifierModel()

        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert "hack_prob" in output
        assert "misalign_prob" in output
        assert output["hack_prob"].shape == (batch_size,)
        assert output["misalign_prob"].shape == (batch_size,)

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_output_range(self, mock_auto_model):
        """Test that output probabilities are in [0, 1]."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 5, 768)
        mock_encoder.return_value = mock_output

        mock_auto_model.from_pretrained.return_value = mock_encoder

        model = CoTClassifierModel()

        input_ids = torch.randint(0, 1000, (1, 5))
        attention_mask = torch.ones(1, 5)

        output = model(input_ids=input_ids, attention_mask=attention_mask)

        assert 0 <= output["hack_prob"].item() <= 1
        assert 0 <= output["misalign_prob"].item() <= 1


class TestMLDetector:
    """Tests for MLDetector."""

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_initialization(self, mock_auto_model, mock_tokenizer):
        """Test detector initialization."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        detector = MLDetector(device="cpu")
        assert detector.name == "ml_detector"
        assert detector.hack_threshold == 0.5
        assert detector.misalign_threshold == 0.5
        assert detector.device.type == "cpu"

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_custom_thresholds(self, mock_auto_model, mock_tokenizer):
        """Test detector with custom thresholds."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        detector = MLDetector(device="cpu", hack_threshold=0.7, misalign_threshold=0.8)
        assert detector.hack_threshold == 0.7
        assert detector.misalign_threshold == 0.8

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_device_auto_detection_cpu(self, mock_auto_model, mock_tokenizer):
        """Test device auto-detection falls back to CPU."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        with patch("torch.backends.mps.is_available", return_value=False):
            with patch("torch.cuda.is_available", return_value=False):
                detector = MLDetector()
                assert detector.device.type == "cpu"

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_extract_texts_cot_traces(self, mock_auto_model, mock_tokenizer):
        """Test extracting texts from cot_traces."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        detector = MLDetector(device="cpu")
        trajectory = {"cot_traces": ["Trace 1", "Trace 2", ""]}

        texts = detector._extract_texts(trajectory)
        assert len(texts) == 2  # Empty trace should be skipped
        assert texts[0] == ("cot_trace_0", "Trace 1")
        assert texts[1] == ("cot_trace_1", "Trace 2")

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_extract_texts_steps(self, mock_auto_model, mock_tokenizer):
        """Test extracting texts from steps with reasoning."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        detector = MLDetector(device="cpu")
        trajectory = {
            "steps": [
                {"thought": "I'm thinking...", "action": "test"},
                {"reasoning": "Let me reason about this"},
                {"cot": "Chain of thought here"},
            ]
        }

        texts = detector._extract_texts(trajectory)
        assert len(texts) == 3
        assert any("I'm thinking" in t[1] for t in texts)
        assert any("Let me reason" in t[1] for t in texts)
        assert any("Chain of thought" in t[1] for t in texts)

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_extract_texts_text_field(self, mock_auto_model, mock_tokenizer):
        """Test extracting texts from text field."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        detector = MLDetector(device="cpu")
        trajectory = {"text": "Direct text content"}

        texts = detector._extract_texts(trajectory)
        assert len(texts) == 1
        assert texts[0] == ("text", "Direct text content")

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_prob_to_risk_levels(self, mock_auto_model, mock_tokenizer):
        """Test probability to risk level conversion."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        detector = MLDetector(device="cpu")

        assert detector._prob_to_risk(0.95) == RiskLevel.CRITICAL
        assert detector._prob_to_risk(0.8) == RiskLevel.HIGH
        assert detector._prob_to_risk(0.6) == RiskLevel.MEDIUM
        assert detector._prob_to_risk(0.3) == RiskLevel.LOW
        assert detector._prob_to_risk(0.1) == RiskLevel.NONE

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_detect_empty_trajectory(self, mock_auto_model, mock_tokenizer):
        """Test detection on empty trajectory."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        detector = MLDetector(device="cpu")
        result = detector.detect({})

        assert result.score == 0.0
        assert result.risk_level == RiskLevel.NONE
        assert len(result.detections) == 0
        assert result.metadata["texts_analyzed"] == 0

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_predict_single(self, mock_auto_model, mock_tokenizer):
        """Test single text prediction."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768

        # Mock forward pass
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        mock_encoder.return_value = mock_output

        mock_auto_model.from_pretrained.return_value = mock_encoder

        # Mock tokenizer
        mock_tokenizer.from_pretrained.return_value.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }

        detector = MLDetector(device="cpu")

        # Mock the model's forward pass
        with patch.object(detector.model, "forward") as mock_forward:
            mock_forward.return_value = {
                "hack_prob": torch.tensor([0.7]),
                "misalign_prob": torch.tensor([0.3]),
            }

            result = detector.predict("Test text")
            assert "hack_prob" in result
            assert "misalign_prob" in result

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_predict_batch_empty(self, mock_auto_model, mock_tokenizer):
        """Test batch prediction with empty list."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        detector = MLDetector(device="cpu")
        result = detector.predict_batch([])

        assert result == []

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_load_weights_not_found(self, mock_auto_model, mock_tokenizer):
        """Test loading weights from non-existent file."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_auto_model.from_pretrained.return_value = mock_encoder

        with pytest.raises(FileNotFoundError, match="Model weights not found"):
            MLDetector(device="cpu", model_path="/nonexistent/model.pt")


class TestMLDetectorIntegration:
    """Integration tests for MLDetector."""

    @patch("rewardhackwatch.core.detectors.ml_detector.AutoTokenizer")
    @patch("rewardhackwatch.core.detectors.ml_detector.AutoModel")
    def test_detect_with_trajectory(self, mock_auto_model, mock_tokenizer):
        """Test full detection pipeline with trajectory."""
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 10, 768)
        mock_encoder.return_value = mock_output

        mock_auto_model.from_pretrained.return_value = mock_encoder

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        detector = MLDetector(device="cpu", hack_threshold=0.3)

        # Mock model forward
        with patch.object(detector.model, "forward") as mock_forward:
            mock_forward.return_value = {
                "hack_prob": torch.tensor([0.6, 0.7]),
                "misalign_prob": torch.tensor([0.4, 0.5]),
            }

            trajectory = {
                "cot_traces": ["Let me hack this", "I'll bypass the test"],
            }

            result = detector.detect(trajectory)

            assert result.detector_name == "ml_detector"
            assert result.score > 0
            assert result.metadata["texts_analyzed"] == 2
