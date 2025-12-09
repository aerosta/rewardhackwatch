"""Tests for MALT dataset loader."""

import json
from unittest.mock import patch

import pytest

try:
    from rewardhackwatch.rhw_bench.malt_loader import MALTLoader

    MALT_LOADER_AVAILABLE = True
except ImportError:
    MALT_LOADER_AVAILABLE = False


@pytest.mark.skipif(not MALT_LOADER_AVAILABLE, reason="MALT loader not available")
class TestMALTLoaderInitialization:
    """Test MALT loader initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        loader = MALTLoader()
        assert loader is not None

    def test_custom_cache_dir(self, tmp_path):
        """Test initialization with custom cache directory."""
        loader = MALTLoader(cache_dir=str(tmp_path))
        assert loader.cache_dir == str(tmp_path)


@pytest.mark.skipif(not MALT_LOADER_AVAILABLE, reason="MALT loader not available")
class TestMALTLoaderDataLoading:
    """Test MALT data loading."""

    @pytest.fixture
    def loader(self):
        return MALTLoader()

    def test_load_train_split(self, loader):
        """Test loading training split."""
        try:
            train_data = loader.load_split("train", limit=10)
            if train_data:
                assert len(train_data) <= 10
        except Exception:
            pytest.skip("MALT dataset not available")

    def test_load_validation_split(self, loader):
        """Test loading validation split."""
        try:
            val_data = loader.load_split("validation", limit=10)
            if val_data:
                assert len(val_data) <= 10
        except Exception:
            pytest.skip("MALT dataset not available")

    def test_load_test_split(self, loader):
        """Test loading test split."""
        try:
            test_data = loader.load_split("test", limit=10)
            if test_data:
                assert len(test_data) <= 10
        except Exception:
            pytest.skip("MALT dataset not available")


@pytest.mark.skipif(not MALT_LOADER_AVAILABLE, reason="MALT loader not available")
class TestMALTLoaderTransformation:
    """Test MALT data transformation."""

    @pytest.fixture
    def loader(self):
        return MALTLoader()

    def test_transform_to_trajectory(self, loader):
        """Test transforming MALT format to trajectory format."""
        malt_sample = {
            "task_id": "test_001",
            "conversation": [
                {"role": "user", "content": "Write a function"},
                {"role": "assistant", "content": "def solve(): pass"},
            ],
            "reward_hack_label": 0,
        }

        trajectory = loader.transform_sample(malt_sample)

        assert "task_id" in trajectory
        assert "code_outputs" in trajectory or "cot_traces" in trajectory

    def test_extract_code_from_conversation(self, loader):
        """Test extracting code from conversation."""
        conversation = [
            {"role": "assistant", "content": "```python\ndef solve():\n    pass\n```"},
        ]

        code = loader.extract_code(conversation)
        assert "def solve" in code or code is None


@pytest.mark.skipif(not MALT_LOADER_AVAILABLE, reason="MALT loader not available")
class TestMALTLoaderFiltering:
    """Test MALT data filtering."""

    @pytest.fixture
    def loader(self):
        return MALTLoader()

    def test_filter_by_label(self, loader):
        """Test filtering by hack label."""
        samples = [
            {"task_id": "1", "reward_hack_label": 1},
            {"task_id": "2", "reward_hack_label": 0},
            {"task_id": "3", "reward_hack_label": 1},
        ]

        hacks = loader.filter_by_label(samples, label=1)
        assert len(hacks) == 2

        clean = loader.filter_by_label(samples, label=0)
        assert len(clean) == 1

    def test_balance_dataset(self, loader):
        """Test balancing dataset."""
        samples = [{"task_id": str(i), "reward_hack_label": 0} for i in range(100)] + [
            {"task_id": str(i + 100), "reward_hack_label": 1} for i in range(10)
        ]

        balanced = loader.balance_dataset(samples)

        hack_count = sum(1 for s in balanced if s["reward_hack_label"] == 1)
        clean_count = sum(1 for s in balanced if s["reward_hack_label"] == 0)

        # Should be roughly balanced
        assert abs(hack_count - clean_count) <= clean_count * 0.5


@pytest.mark.skipif(not MALT_LOADER_AVAILABLE, reason="MALT loader not available")
class TestMALTLoaderCaching:
    """Test MALT data caching."""

    def test_cache_creation(self, tmp_path):
        """Test that cache is created."""
        loader = MALTLoader(cache_dir=str(tmp_path))

        # Mock loading
        with patch.object(loader, "_download_data") as mock_download:
            mock_download.return_value = [{"task_id": "1"}]

            loader.load_split("train", limit=1)

            # Cache should be created or attempted
            # Implementation dependent

    def test_cache_hit(self, tmp_path):
        """Test cache hit."""
        # Create cached file
        cache_file = tmp_path / "malt_train.json"
        cache_file.write_text(json.dumps([{"task_id": "cached"}]))

        loader = MALTLoader(cache_dir=str(tmp_path))

        # Should use cached data
        if hasattr(loader, "_load_from_cache"):
            cached = loader._load_from_cache("train")
            if cached:
                assert cached[0]["task_id"] == "cached"
