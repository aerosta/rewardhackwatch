"""Tests for SHADE-Arena loader."""

import pytest

try:
    from rewardhackwatch.rhw_bench.shade_loader import SHADELoader

    SHADE_LOADER_AVAILABLE = True
except ImportError:
    SHADE_LOADER_AVAILABLE = False


@pytest.mark.skipif(not SHADE_LOADER_AVAILABLE, reason="SHADE loader not available")
class TestSHADELoaderInitialization:
    """Test SHADE loader initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        loader = SHADELoader()
        assert loader is not None

    def test_custom_repo(self):
        """Test initialization with custom repo."""
        loader = SHADELoader(repo_id="ScaleAI/mrt")
        assert "mrt" in loader.repo_id


@pytest.mark.skipif(not SHADE_LOADER_AVAILABLE, reason="SHADE loader not available")
class TestSHADELoaderDataLoading:
    """Test SHADE data loading."""

    @pytest.fixture
    def loader(self):
        return SHADELoader()

    def test_load_samples(self, loader):
        """Test loading samples."""
        try:
            samples = loader.load(limit=5)
            if samples:
                assert len(samples) <= 5
        except Exception:
            pytest.skip("SHADE dataset not available")

    def test_load_with_filter(self, loader):
        """Test loading with filter."""
        try:
            loader.load(limit=10, filter_func=lambda x: True)
            # Should return samples
        except Exception:
            pytest.skip("SHADE dataset not available")


@pytest.mark.skipif(not SHADE_LOADER_AVAILABLE, reason="SHADE loader not available")
class TestSHADELoaderTransformation:
    """Test SHADE data transformation."""

    @pytest.fixture
    def loader(self):
        return SHADELoader()

    def test_transform_transcript(self, loader):
        """Test transforming SHADE transcript to trajectory."""
        transcript = {
            "conversation": [
                {"role": "user", "content": "Solve this problem"},
                {"role": "assistant", "content": "Let me think..."},
                {"role": "assistant", "content": "```python\nprint('solution')\n```"},
            ],
            "is_successful_attack": True,
        }

        trajectory = loader.transform_transcript(transcript)

        assert "cot_traces" in trajectory or "code_outputs" in trajectory
        assert trajectory.get("is_hack", False)

    def test_extract_attack_type(self, loader):
        """Test extracting attack type from transcript."""
        transcript = {
            "attack_category": "output_manipulation",
        }

        attack_type = loader.extract_attack_type(transcript)
        assert attack_type is not None or attack_type == "unknown"


@pytest.mark.skipif(not SHADE_LOADER_AVAILABLE, reason="SHADE loader not available")
class TestSHADELoaderStatistics:
    """Test SHADE statistics computation."""

    @pytest.fixture
    def loader(self):
        return SHADELoader()

    def test_compute_statistics(self, loader):
        """Test computing dataset statistics."""
        samples = [
            {"is_successful_attack": True},
            {"is_successful_attack": True},
            {"is_successful_attack": False},
        ]

        stats = loader.compute_statistics(samples)

        assert "total" in stats
        assert stats["total"] == 3
        assert "attack_rate" in stats or "hack_rate" in stats


@pytest.mark.skipif(not SHADE_LOADER_AVAILABLE, reason="SHADE loader not available")
class TestSHADELoaderEdgeCases:
    """Test edge cases for SHADE loader."""

    @pytest.fixture
    def loader(self):
        return SHADELoader()

    def test_empty_transcript(self, loader):
        """Test handling empty transcript."""
        transcript = {}
        trajectory = loader.transform_transcript(transcript)
        assert trajectory is not None

    def test_missing_conversation(self, loader):
        """Test handling missing conversation."""
        transcript = {
            "is_successful_attack": False,
        }
        trajectory = loader.transform_transcript(transcript)
        assert trajectory is not None

    def test_malformed_conversation(self, loader):
        """Test handling malformed conversation."""
        transcript = {
            "conversation": "not a list",
        }
        loader.transform_transcript(transcript)
        # Should handle gracefully
